#!/usr/bin/env python3
"""Rewrite the TOSA that tosa-converter-for-tflite emits into a form that
torq-compile / the Torq runtime can actually execute on the Coral NPU.

Three classes of problems are fixed:

1. i1 tensors.  Any i1 value that reaches the Torq backend crashes the runtime
   with `xram_write: acessing memory out of bounds`.  Every i1 producing op is
   rewritten into f32 arithmetic that yields 0.0/1.0.

2. i32 reciprocal/floor.  `FLOOR_MOD(pos, 4096)` is mistranslated into
   `reciprocal(4096) * pos` (integer reciprocal == 0) then floor/sub, which is
   always 0.  It is rewritten as an f32 floor based modulo.

3. `tfl.dynamic_update_slice` (KV-cache writes) and `tfl.embedding_lookup`,
   which the TOSA lowering does not implement.  The slice update is rewritten
   as a broadcast blend; when the update is wider than one element a shift
   matrix is built and applied with a matmul.

Usage:  tosa_patch.py in.mlir out.mlir [--function fn_4]
"""

import argparse
import re
import struct
import sys

RESULT_RE = re.compile(r'^\s*(%[A-Za-z0-9_]+) = ')
VAL_RE = re.compile(r'%[A-Za-z0-9_]+')
TENSOR_RE = re.compile(r'tensor<([0-9]+(?:x[0-9]+)*)x([a-z0-9]+)>')


def parse_tensor(t):
    m = TENSOR_RE.fullmatch(t or "")
    if not m:
        return None, None
    return [int(v) for v in m.group(1).split("x")], m.group(2)


def tname(shape, dtype):
    return "tensor<%sx%s>" % ("x".join(str(v) for v in shape), dtype)


def bcast(a, b):
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + list(a)
    b = [1] * (n - len(b)) + list(b)
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise SystemExit("shapes not broadcastable: %r %r" % (a, b))
    return out


def hex_f32(values):
    return "hex:" + b"".join(struct.pack("<f", float(v)) for v in values).hex().upper()


class Em:
    """Line emitter with SSA type tracking."""

    def __init__(self):
        self.lines = []
        self.n = 0
        self.cache = {}
        self.types = {}

    def fresh(self):
        self.n += 1
        return "%%p%d" % self.n

    def raw(self, line):
        self.lines.append(line)

    def add(self, text):
        self.lines.append("    " + text)

    # -- constants ---------------------------------------------------------
    def const(self, shape, dtype, text, key=None):
        key = key if key is not None else (tuple(shape), dtype, text)
        if key in self.cache:
            return self.cache[key]
        nm = self.fresh()
        t = tname(shape, dtype)
        if text.startswith("hex:"):
            dense = 'dense<"0x%s"> : %s' % (text[4:], t)
        else:
            dense = "dense<%s> : %s" % (text, t)
        self.add('%s = "tosa.const"() <{values = %s}> : () -> %s' % (nm, dense, t))
        self.types[nm] = t
        self.cache[key] = nm
        return nm

    def splat(self, rank, value):
        txt = "1.000000e+00" if value == 1 else "0.000000e+00"
        return self.const([1] * rank, "f32", txt)

    def shape_const(self, shape):
        key = ("shape", tuple(shape))
        if key in self.cache:
            return self.cache[key]
        nm = self.fresh()
        vals = ", ".join(str(v) for v in shape)
        self.add(
            "%s = tosa.const_shape  {values = dense<[%s]> : tensor<%dxindex>} : () -> !tosa.shape<%d>"
            % (nm, vals, len(shape), len(shape))
        )
        self.types[nm] = "!tosa.shape<%d>" % len(shape)
        self.cache[key] = nm
        return nm

    # -- ops ---------------------------------------------------------------
    def reshape(self, val, shape, out=None):
        s = self.shape_const(shape)
        o = out or self.fresh()
        dims = [0]
        src = self.types[val]
        sshape, sdtype = parse_tensor(src)
        dst = tname(shape, sdtype)
        self.add(
            "%s = tosa.reshape %s, %s : (%s, !tosa.shape<%d>) -> %s"
            % (o, val, s, src, len(shape), dst)
        )
        self.types[o] = dst
        return o

    def cast(self, val, dtype, out=None):
        o = out or self.fresh()
        src = self.types[val]
        sshape, _ = parse_tensor(src)
        dst = tname(sshape, dtype)
        self.add("%s = tosa.cast %s : (%s) -> %s" % (o, val, src, dst))
        self.types[o] = dst
        return o

    def unop(self, op, a, out=None):
        o = out or self.fresh()
        ta = self.types[a]
        self.add("%s = tosa.%s %s : (%s) -> %s" % (o, op, a, ta, ta))
        self.types[o] = ta
        return o

    def mul(self, a, b, out=None):
        sh = self.const([1], "i8", "0", key=("shift",))
        o = out or self.fresh()
        ta, tb = self.types[a], self.types[b]
        sa, da = parse_tensor(ta)
        sb, _ = parse_tensor(tb)
        r = bcast(sa, sb)
        t = tname(r, da)
        self.add("%s = tosa.mul %s, %s, %s : (%s, %s, %s) -> %s"
                 % (o, a, b, sh, ta, tb, self.types[sh], t))
        self.types[o] = t
        return o

    def binop(self, op, a, b, out=None, attrs=""):
        o = out or self.fresh()
        ta, tb = self.types[a], self.types[b]
        sa, da = parse_tensor(ta)
        sb, _ = parse_tensor(tb)
        r = bcast(sa, sb)
        t = tname(r, da)
        self.add("%s = tosa.%s %s, %s%s : (%s, %s) -> %s" % (o, op, a, b, attrs, ta, tb, t))
        self.types[o] = t
        return o


def build_types(lines):
    types = {}
    for line in lines:
        m = RESULT_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if " -> " in line:
            t = line.rsplit(" -> ", 1)[1].strip()
            types[name] = t
    if lines and lines[0].lstrip().startswith("func.func @"):
        for m in re.finditer(r'(%[A-Za-z0-9_]+): (tensor<[^>]*>)', lines[0]):
            types[m.group(1)] = m.group(2)
    return types


def find_floormod(em, lines, types):
    """Locate the `reciprocal(i32)` FLOOR_MOD chain.  Returns (start, end, result)."""
    for i, line in enumerate(lines):
        if "tosa.reciprocal" not in line:
            continue
        ops = VAL_RE.findall(line)
        if len(ops) < 2:
            continue
        operand = ops[1]
        t = types.get(operand, "")
        if not t.endswith("xi32>"):
            continue
        # reciprocal -> mul -> floor -> sub; the FLOOR_MOD result is the sub
        res = VAL_RE.findall(lines[i + 3])[0]
        return i, i + 3, res
    return None


def find_dus(lines, types):
    out = []
    for i, line in enumerate(lines):
        if "tfl.dynamic_update_slice" not in line:
            continue
        ops = VAL_RE.findall(line)
        name = ops[0]
        base, upd, idx = ops[1], ops[2], ops[3]
        bs, _ = parse_tensor(types[base])
        us, _ = parse_tensor(types[upd])
        if len(bs) != len(us):
            raise SystemExit("dus rank mismatch %r %r" % (bs, us))
        diffs = [k for k in range(len(bs)) if bs[k] != us[k]]
        if len(diffs) != 1:
            raise SystemExit("dus: ambiguous axis %r %r" % (bs, us))
        ax = diffs[0]
        out.append((i, name, base, upd, ax, bs[ax], us[ax]))
    return out


def rewrite(lines, fn_name, embedding=None):
    types = build_types(lines)
    em = Em()
    em.types.update(types)
    if embedding is not None:
        em.types[embedding[0]] = embedding[1]

    fm = find_floormod(em, lines, types)
    dus = find_dus(lines, types)
    if fm is None:
        if dus:
            # The KV-cache update consumes the module result and its shift
            # matrix is built from the position modulo; without the chain
            # there is no legal index to shift with.
            raise SystemExit(
                "no i32 FLOOR_MOD chain but %d dynamic_update_slice op(s) "
                "found in %s; cannot rewrite the KV update" % (len(dus), fn_name)
            )
        # Functions without position arithmetic (e.g. the tiny index
        # validators the plugin splits out) need only the i1/general rewrites.
        fm_start = fm_end = -1
        fm_res = None
        print("note: %s: no i32 FLOOR_MOD chain, position op skipped" % fn_name)
    else:
        fm_start, fm_end, fm_res = fm

    if fm is None:
        W = NB = None
    else:
        if not dus:
            raise SystemExit("no dynamic_update_slice found in %s" % fn_name)
        ws = {d[6] for d in dus}
        ns = {d[5] for d in dus}
        if len(ws) != 1 or len(ns) != 1:
            raise SystemExit("dus: mixed widths %r / extents %r" % (ws, ns))
        W = ws.pop()
        NB = ns.pop()

    # state for the lazily emitted shift-matrix block
    shift = {}

    def ensure_shift(idx_val):
        if shift:
            return shift
        rng = em.const([1, NB], "f32", hex_f32(range(NB)))
        js = em.const([W, 1], "f32", hex_f32(range(W)))
        idxf = em.cast(idx_val, "f32")
        t1 = em.binop("sub", rng, idxf)
        d = em.binop("sub", t1, js)
        ad = em.unop("abs", d)
        one2 = em.splat(2, 1)
        mn = em.binop("minimum", ad, one2)
        M = em.binop("sub", one2, mn)
        mm = em.fresh()
        em.add("%s = tosa.reduce_sum %s {axis = 0 : i32} : (%s) -> %s"
               % (mm, M, em.types[M], tname([1, NB], "f32")))
        em.types[mm] = tname([1, NB], "f32")
        zpt = em.const([1], "f32", "0.000000e+00")
        shift.update(rng=rng, js=js, idxf=idxf, M=M, mm=mm, zpt=zpt)
        return shift

    def emit_dus(out, base, upd, ax):
        sh = ensure_shift(fm_res2)
        bs, _ = parse_tensor(types[base])
        us, _ = parse_tensor(types[upd])
        if W == 1:
            mshape = [1, 1, 1, 1]
            mshape[ax] = NB
            mm4 = em.reshape(sh["mm"], mshape)
            d = em.binop("sub", upd, base)
            e = em.mul(d, mm4)
            em.binop("add", e, base, out=out)
            return
        if ax == 3:
            u3 = em.reshape(upd, [1, bs[2], W])
            m3 = em.reshape(sh["M"], [1, W, NB])
            p = matmul(u3, m3)
            p4 = em.reshape(p, bs)
            mm4 = em.reshape(sh["mm"], [1, 1, 1, NB])
        else:
            mt = transp(sh["M"], [1, 0])
            m3 = em.reshape(mt, [1, NB, W])
            u3 = em.reshape(upd, [1, W, bs[3]])
            p = matmul(m3, u3)
            p4 = em.reshape(p, bs)
            mm4 = em.reshape(sh["mm"], [1, 1, NB, 1])
        d = em.binop("sub", p4, base)
        e = em.mul(d, mm4)
        em.binop("add", e, base, out=out)

    def matmul(a, b):
        o = em.fresh()
        ta, tb = em.types[a], em.types[b]
        sa, da = parse_tensor(ta)
        sb, _ = parse_tensor(tb)
        r = [sa[0], sa[1], sb[2]]
        t = tname(r, da)
        z = em.splat(1, 0)
        em.add("%s = tosa.matmul %s, %s, %s, %s : (%s, %s, %s, %s) -> %s"
               % (o, a, b, z, z, ta, tb, em.types[z], em.types[z], t))
        em.types[o] = t
        return o

    def transp(a, perms):
        o = em.fresh()
        ta = em.types[a]
        sa, da = parse_tensor(ta)
        r = [sa[p] for p in perms]
        t = tname(r, da)
        em.add("%s = tosa.transpose %s {perms = array<i32: %s>} : (%s) -> %s"
               % (o, a, ", ".join(str(p) for p in perms), ta, t))
        em.types[o] = t
        return o

    fm_res2 = fm_res  # value produced by the rewritten FLOOR_MOD

    # f32 0/1 comparison -> replaces an i1 producing op
    def emit_cmp(op, out, a, b):
        ta, tb = em.types[a], em.types[b]
        sa, da = parse_tensor(ta)
        sb, db = parse_tensor(tb)
        r = bcast(sa, sb)
        rank = len(r)
        if da != "f32":
            a = em.cast(a, "f32")
            ta = em.types[a]
        if db != "f32":
            b = em.cast(b, "f32")
            tb = em.types[b]
        if len(sa) != rank:
            a = em.reshape(a, r)
        if len(sb) != rank:
            b = em.reshape(b, r)
        one = em.splat(rank, 1)
        zero = em.splat(rank, 0)
        if op == "greater_equal":
            s = em.binop("add", a, one)
        else:
            s = a
        d = em.binop("sub", s, b)
        c = em.binop("maximum", d, zero)
        em.binop("minimum", c, one, out=out)

    index = 0
    n = len(lines)
    # the shift block must be inserted before its first use, which is the first
    # DUS; the FLOOR_MOD result is available well before that.
    while index < n:
        line = lines[index]
        stripped = line.strip()
        m = RESULT_RE.match(line)
        if not m:
            em.raw(line.rstrip("\n"))
            index += 1
            continue
        name = m.group(1)
        ops = VAL_RE.findall(line)

        # --- FLOOR_MOD ---------------------------------------------------
        if index == fm_start:
            mul = lines[index + 1]
            subs = lines[index + 3]
            x = VAL_RE.findall(mul)[1]          # pos
            r = VAL_RE.findall(subs)[0]         # result ssa (== fm_res)
            xt = em.types[x]
            dt = em.cast(x, "f32")
            inv = em.const([1, 1], "f32", "%.9e" % (1.0 / NB))
            full = em.const([1, 1], "f32", "%.9e" % float(NB))
            q = em.mul(dt, inv)
            fl = em.fresh()
            em.add("%s = tosa.floor %s : (%s) -> %s" % (fl, q, em.types[q], em.types[q]))
            em.types[fl] = em.types[q]
            t2 = em.mul(fl, full)
            rr = em.binop("sub", dt, t2)
            em.cast(rr, "i32", out=r)
            index += 4
            continue

        # --- i1 comparisons ----------------------------------------------
        if stripped.split("=")[1].strip().startswith(("tosa.greater_equal", "tosa.greater ", "tosa.less ", "tosa.equal ")):
            op = re.search(r"tosa\.([a-z_]+)", stripped).group(1)
            emit_cmp(op, name, ops[1], ops[2])
            index += 1
            continue

        if stripped.split("=")[1].strip().startswith("tosa.logical_and"):
            em.mul(ops[1], ops[2], out=name)
            index += 1
            continue

        if stripped.split("=")[1].strip().startswith("tosa.logical_not"):
            a = ops[1]
            sa, _ = parse_tensor(em.types[a])
            one = em.splat(len(sa), 1)
            em.binop("sub", one, a, out=name)
            index += 1
            continue

        if stripped.split("=")[1].strip().startswith("tosa.cast") and "xi1>" in stripped:
            a = ops[1]
            rshape, _ = parse_tensor(em.types[name])
            zero = em.splat(len(rshape), 0)
            em.binop("add", a, zero, out=name)
            index += 1
            continue

        if stripped.split("=")[1].strip().startswith("tosa.select"):
            mask, x, y = ops[1], ops[2], ops[3]
            tx, ty = em.types[x], em.types[y]
            xf = em.cast(x, "f32") if not tx.endswith("xf32>") else x
            yf = em.cast(y, "f32") if not ty.endswith("xf32>") else y
            d = em.binop("sub", xf, yf)
            p = em.mul(d, mask)
            s = em.binop("add", p, yf)
            em.cast(s, "i32", out=name)
            index += 1
            continue

        # --- KV cache update ----------------------------------------------
        dus_entry = next((d for d in dus if d[0] == index), None)
        if dus_entry is not None and fm is not None:
            _, out, base, upd, ax, _, _ = dus_entry
            emit_dus(out, base, upd, ax)
            index += 1
            continue

        if "tosa.conv2d" in stripped:
            mm = re.match(
                r"^(%[A-Za-z0-9_]+) = tosa\.conv2d (%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+), "
                r"(%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+) (\{.*\}) : "
                r"\((.*)\) -> (.*)$", stripped)
            if mm is None:
                raise SystemExit("cannot parse conv2d: %s" % stripped[:120])
            op1, op2, op3, op4, op5 = (mm.group(k) for k in range(2, 7))
            attrs, tout = mm.group(7), mm.group(9)
            m1 = re.fullmatch(r"tensor<(\d+)x1x1x(\d+)xf32>", types[op1])
            m2 = re.fullmatch(r"tensor<(\d+)x1x1x(\d+)xf32>", tout)
            if m1 and m2 and int(m1.group(1)) > 1:
                # A 1x1 conv2d over `T` tokens is a (T x K) * (K x C) matmul, and
                # Torq's conv2d lowering gets the layout of its result wrong as
                # soon as the consumer wants it transposed (which the attention
                # path does).  Emit the matmul instead: same arithmetic, and the
                # matmul path is the one that is exercised end to end.
                T, K = int(m1.group(1)), int(m1.group(2))
                C = int(m2.group(2))
                a = em.reshape(op1, [1, T, K])
                wt = transp(op2, [1, 2, 3, 0])
                b = em.reshape(wt, [1, K, C])
                p = matmul(a, b)
                em.reshape(p, [T, 1, 1, C], out=name)
                index += 1
                continue
            em.raw(line.rstrip("\n"))
            index += 1
            continue

        if "tfl.embedding_lookup" in stripped:
            if embedding is None:
                raise SystemExit(
                    "tfl.embedding_lookup present; exclude EMBEDDING_LOOKUP from "
                    "the NPU partition or pass --embedding-arg"
                )
            rshape, _ = parse_tensor(em.types[name])
            zero = em.splat(len(rshape), 0)
            em.binop("add", embedding[0], zero, out=name)
            index += 1
            continue

        em.raw(line.rstrip("\n"))
        index += 1

    return em.lines


def stream_functions(fh):
    """Yield ("text", line) for preamble lines and ("func", [lines]) per function.

    Functions from tosa-converter-for-tflite are flat: the body is 4-space
    indented and the closing brace sits at 2 spaces, so no real parser is
    needed and a multi-gigabyte module can be processed a function at a time.
    """
    body = None
    for line in fh:
        stripped = line.rstrip("\n")
        if body is None:
            if stripped.startswith("  func.func @"):
                body = [stripped]
            else:
                yield "text", stripped
            continue
        body.append(stripped)
        if stripped == "  }":
            yield "func", body
            body = None
    if body is not None:
        yield "func", body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--embedding-arg", default=None,
                    help="%%argN to substitute for tfl.embedding_lookup (bring-up only)")
    ap.add_argument("--rename", default=None,
                    help="OLD:NEW function name prefix, e.g. fn_:partition_")
    ap.add_argument("--keep-attrs", action="store_true",
                    help="keep tf_saved_model.* attributes (they are stripped by default)")
    args = ap.parse_args()

    rename = None
    if args.rename:
        old, _, new = args.rename.partition(":")
        rename = (old, new)

    written = 0
    with open(args.src) as fh, open(args.dst, "w") as out:
        for kind, payload in stream_functions(fh):
            if kind == "text":
                text = payload
                if rename and ("@" + rename[0]) in text:
                    text = text.replace("@" + rename[0], "@" + rename[1])
                out.write(text + "\n")
                written += 1
                continue

            body = payload
            head = body[0]
            name = head[head.index("@") + 1:head.index("(")]
            if rename and name.startswith(rename[0]):
                name = rename[1] + name[len(rename[0]):]
                body[0] = head.replace("@" + head[head.index("@") + 1:head.index("(")],
                                       "@" + name, 1)
            elif rename:
                raise SystemExit("unexpected function name %r" % name)

            if not args.keep_attrs:
                def _strip(b):
                    b = re.sub(r' \{tf_saved_model[^}]*\}', "", b)
                    k = b.find(" attributes {")
                    return b[:k] + " {" if k >= 0 else b
                body = [_strip(b) for b in body]

            if args.embedding_arg is not None:
                rshape = None
                for b in body:
                    if "tfl.embedding_lookup" in b:
                        rshape = re.search(r"-> (tensor<[^>]*>)", b).group(1)
                if rshape is None:
                    raise SystemExit("no embedding_lookup to substitute")
                head = body[0]
                lp = head.index("(")
                depth = 0
                for k in range(lp, len(head)):
                    if head[k] == "(":
                        depth += 1
                    elif head[k] == ")":
                        depth -= 1
                        if depth == 0:
                            rp = k
                            break
                else:
                    raise SystemExit("unbalanced func signature")
                body[0] = (head[:rp] + ", %s: %s" % (args.embedding_arg, rshape)
                           + head[rp:])
                body = rewrite(body, name, (args.embedding_arg, rshape))
            else:
                body = rewrite(body, name)

            out.write("\n".join(body) + "\n")
            written += len(body)
    print("wrote %s (%d lines)" % (args.dst, written))


if __name__ == "__main__":
    main()
