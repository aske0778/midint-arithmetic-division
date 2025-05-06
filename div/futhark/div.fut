import "div-helpers"
import "big-add"
import "sqr-mul"
import "futhark_new/sub"
import "futhark_new/mul"


let us = [1,1,1,1] :> [1*(4*1)]u16
let vs = [2,2,2,2] :> [1*(4*1)]u16

--
-- Calculates (a * b) rem B^d
--
def multmod [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (d: i64) : [ipb*(4*m)]u16 = 
    let res = convMulV2 us vs
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0u16 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (l: i64) : (u32, [ipb*(4*m)]u16) =
    let precV = prec vs
    let precW = prec ws
    let L = precW + precV - l + 1

    in if (precV == 0 || precW == 0) then
        let ret = zeroAndSet 1u16 h (ipb*(4*m))
        let ret = ret :> [ipb*(4*m)]u16
        in (1, ret)
    else if (L >= h) then
        let ret = convMulV2 vs ws
        in if ltBpow ret h then
            let bpow = zeroAndSet 1 h (ipb*(4*m))
            let (ret, _) = bsub bpow ret
            in (1, ret)
        else
            let bpow = zeroAndSet 1 h (ipb*(4*m))
            let (ret, _) = bsub ret bpow
            in (0, ret)
    else 
        let ret = multmod vs ws L
        in if !(ez ret) && ret[L-1] == 0 then 
            (0, ret)
        else 
            let bpow = zeroAndSet 1 L (ipb*(4*m))
            let (ret, _) = bsub bpow ret
            in (1, ret)

--
-- Iterate towards an approximation in at most log(M) steps
--
def step [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (l: i64) (n: i64) : [ipb*(4*m)]u16 =
    let (sign, tmp) = powDiff vs ws (h-n) (l-2)
    let tmp = convMulV2 ws tmp
        |> shift (2 * n - h)
    let ws = shift n ws
    in if sign != 0 then
        baddu16 ws tmp
    else
        bsubu16 ws tmp

--
-- Refine the approximation of the quotient
--
def refine [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (k: i64) (l: i64) : [ipb*(4*m)]u16 =
    let ws = shift 2 ws
    let (ws, _) = loop (ws, l) = (ws, l)
        while h - k > l do
            let n = i64.min (h - k + 1 - l) l
            let s = i64.max 0 (k - 2 * l + 1 - 2)
            let vs = shift (-s) vs
            let tmp = step vs ws (k + l + n - s + 2) n l
            let ws = shift (-1) tmp
            let l = l + n - 1
            in (ws, l)
    in shift (-2) ws

--
-- Calculates the shifted inverse
--
def shinv [m][ipb] (vs: [ipb*(4*m)]u16) (h: i64) (k: i64) : [ipb*(4*m)]u16 =
    if k == 0 then
        quo_single h (vs) (ipb*(4*m)) :> [ipb*(4*m)]u16 
    else if k >= h && !(eqBpow vs h) then
        vs
    else if k == h - 1 && vs[k] > u16.highest / 2 then
        zeroAndSet 1 0 (ipb*(4*m)) :> [ipb*(4*m)]u16
    else if eqBpow vs k then
        zeroAndSet 1 (h - k) (ipb*(4*m)) :> [ipb*(4*m)]u16
    else 
        let V = (u64.u16 vs[k - 2]) | (u64.u16 vs[k - 1]) << 1*16 | (u64.u16 vs[k]) << 2*16
        let b2l = 1u64 << 4*16
        let tmp = (b2l - V) / V + 1

        let ws = tabulate (ipb*(4*m)) (\i -> 
            if i == 0 then u16.u64 tmp
            else if i == 1 then u16.u64 (tmp >> 16)
            else 0u16 )

        in refine vs ws h k 2

--
-- Implementation of multi-precision integer division using
-- the shifted inverse and classical multiplication
--
def div [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) : ([ipb*(4*m)]u16, [ipb*(4*m)]u16) =
    let h = prec us
    let k = (prec vs) - 1

    let (kIsOne, us, vs, h, k) =
        if k == 1 then
            let h = h + 1
            let k = k + 1
            let us = shift 1 us
            let vs = shift 1 vs
            in (true, us, vs, h, k)
        else
            (false, us, vs, h, k)

    let quo = shinv vs h k
        |> convMulV2 us
        |> shift (-h)

    let (rem, _) = convMulV2 vs quo
        |> bsub us

    let (quo, rem) =
        if lt rem vs then
            let quo = badd1u16 quo
            let (rem, _) = bsub rem vs
            in (quo, rem)
        else
            (quo, rem)

    let rem =
        if kIsOne then
            shift (-1) rem
        else 
            rem

    in (quo, rem)

--
-- Implementation of multi-precision integer quotient using
-- the shifted inverse and classical multiplication
--
def quo [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) : [ipb*(4*m)]u16 =
    let h = prec us
    let k = (prec vs) - 1

    let (us, vs, h, k) =
        if k == 1 then
            let h = h + 1
            let k = k + 1
            let us = shift (-1) us
            let vs = shift (-1) vs
            in (us, vs, h, k)
        else
            (us, vs, h, k)

    let quo = shinv vs h k
        |> bmulu16 us
        |> shift (-h)

    let rem = bmulu16 vs quo
        |> bsubu16 us

    let quo =
        if lt rem vs then
            badd1u16 quo
        else
            quo

    in quo



--
-- ==
-- entry: bench_div
-- compiled random input { [131072][64]u16  [131072][64]u16 }
-- compiled random input { [65536][128]u16  [65536][128]u16 }
-- compiled random input { [8192][256]u16  [8192][256]u16 }
-- compiled random input { [16384][512]u16  [16384][512]u16 }
-- compiled random input { [8192][1024]u16  [8192][1024]u16 }
-- compiled random input { [4096][2048]u16  [4096][2048]u16 }
-- compiled random input { [2048][4096]u16  [2048][4096]u16 }
-- compiled random input { [1024][8192]u16  [1024][8192]u16 }
-- compiled random input { [512][16384]u16  [512][16384]u16 }
entry bench_div [n][m] (us: [n][m]u16) (vs: [n][m]u16) : ([n][m]u16, [n][m]u16) = 
    #[unsafe]
    let mdiv4 = m / 4
    let us = us :> [n][1*(4*mdiv4)]u16
    let vs = vs :> [n][1*(4*mdiv4)]u16
    let ret = map2 div us vs 
    let ret = unzip ret :> ([n][m]u16, [n][m]u16)
    in ret

--
-- ==
-- entry: bench_div64
-- compiled random input { [131072][64]u16  [131072][64]u16 }
entry bench_div64 [n][m] (us: [n][m]u16) (vs: [n][m]u16) : ([n][m]u16, [n][m]u16) = 
    #[unsafe]
    let mdiv4 = m / 4
    let us = us :> [n][1*(4*mdiv4)]u16
    let vs = vs :> [n][1*(4*mdiv4)]u16
    let ret = map2 div us vs 
    let ret = unzip ret :> ([n][m]u16, [n][m]u16)
    in ret




--
-- ==
-- entry: bench_quo
-- compiled random input { [131072][64]u16  [3]u16 }
-- compiled random input { [65536][128]u16  [3]u16 }
-- compiled random input { [32768][256]u16  [3]u16 }
-- compiled random input { [16384][512]u16  [3]u16 }
-- compiled random input { [8192][1024]u16  [3]u16 }
-- compiled random input { [4096][2048]u16  [3]u16 }
-- compiled random input { [2048][4096]u16  [3]u16 }
-- compiled random input { [1024][8192]u16  [3]u16 }
-- compiled random input { [512][16384]u16  [3]u16 }
-- compiled random input { [256][32768]u16  [3]u16 }
entry bench_quo [m][n] (us: [n][m]u16) (vs: [3]u16) : [n][m]u16 =
    let vs = tabulate m (\i -> if i < 3 then vs[i] else 0)
    let mdiv4 = m / 4
    let vs = replicate n vs
    let us = us :> [n][1*(4*mdiv4)]u16
    let vs = vs :> [n][1*(4*mdiv4)]u16
    let ret = imap2Intra us vs quo :> [n][m]u16
    in ret


--
-- ==
-- entry: bench_quo_single
-- compiled random input {  [64]u16  [3]u16 }
-- compiled random input { [128]u16  [3]u16 }
-- compiled random input {  [256]u16  [3]u16 }
-- compiled random input {  [512]u16  [3]u16 }
-- compiled random input { [1024]u16  [3]u16 }
-- compiled random input {  [2048]u16  [3]u16 }
-- compiled random input {  [4096]u16  [3]u16 }
-- compiled random input {  [8192]u16  [3]u16 }
-- compiled random input {  [16384]u16  [3]u16 }
-- compiled random input {  [32768]u16  [3]u16 }
entry bench_quo_single [m] (us: [m]u16) (vs: [3]u16) : [m]u16 =
    let vs = tabulate m (\i -> if i < 3 then vs[i] else 0)
    let mdiv4 = m / 4
    let us = us :> [1*(4*mdiv4)]u16
    let vs = vs :> [1*(4*mdiv4)]u16
    let ret = quo us vs :> [m]u16
    in ret


