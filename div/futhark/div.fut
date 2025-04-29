import "div-helpers"
import "big-add"
import "sqr-mul"

--let us = [1,4,2,3, 0, 0, 0, 0] :> [1*(4*2)]u16
--let vs = [0,0,4,1, 0, 0, 0, 0] :> [1*(4*2)]u16

let us = [1,4,2,3] :> [1*(4*1)]u16
let vs = [0,0,4,1] :> [1*(4*1)]u16

--
-- Calculates (a * b) rem B^d
--
def multmod [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (d: i64) : [ipb*(4*m)]u16 = 
    let res = bmulu16 us vs
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0u16 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (h: i64) (l: i64) : (u32, [ipb*(4*m)]u16) =
    let precU = prec us
    let precV = prec vs
    let L = precV + precU - l + 1

    in if (precU == 0 || precV == 0) then
        let ret = zeroAndSet 1u16 h m
        let ret = ret :> [ipb*(4*m)]u16
        in (1, ret)
    else if (L >= h) then
        let ret = bmulu16 us vs
        in if ltbpow ret h then
            let ret = subbpowbigint h ret
            in (1, ret)
        else
            let ret = subbigintbpow ret h
            in (0, ret)
    else 
        let ret = multmod us vs L
        in if !(ez ret) && ret[L-1] == 0 then 
            (0, ret)
        else 
            let ret = subbpowbigint L ret
            in (1, ret)

--
-- Iterate towards an approximation in at most log(M) steps
--
def step [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (h: i64) (l: i64) (n: i64) : [ipb*(4*m)]u16 =
    let (sign, vs) = powDiff us vs (h-n) (l-2)
    let vs = bmulu16 us vs
        |> shift (2 * n - h)
    let us = shift n us
    in if sign != 0 then
        baddu16 us vs
    else
        bsubu16 us vs

--
-- Refine the approximation of the quotient
--
def refine [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (h: i64) (k: i64) (l: i64) : [ipb*(4*m)]u16 =
    let us = shift 2 us
    let (_, vs, _) = loop (us, vs, l) = (us, vs, l)
        while h - k > l do
            let n = i64.min (h - k + 1 - l) l
            let s = i64.max 0 (k - 2 * l + 1 - 2)
            let us = shift (-s) us
            let us = step us vs (k + l + n - s + 2) n l
            let us = shift (-1) us
            let l = l + n - 1
            in (us, vs, l)
    in shift (-2) vs

def quo_single [m][ipb] (bpow : i64) (d :[ipb*(4*m)]u16) (n : i64) : ([ipb*(4*m)]u16) =
    let ret = replicate n 0u16 :> [ipb*(4*m)]u16
    let (_r,ret) = loop (r,ret) = (1u32, copy ret) for i < bpow do
                let r = r << 16
                in
                if (r > (u32.u16 d[0])) then 
                                let ret[(bpow - 1) - i] = u16.u32(r / (u32.u16 d[0]))
                                in
                                (r % (u32.u16 d[0]), ret)
                        else 
                                (r, ret)
    in ret

--
-- Calculates the shifted inverse
--
def shinv [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (h: i64) (k: i64) : [ipb*(4*m)]u16 =
    if k == 0 then
        quo_single h (vs) (ipb*(4*m)) :> [ipb*(4*m)]u16--map u16.i64 (iota (ipb*(4*m))) -- TODO: implement quo
    else if k >= h && !(eqBpow vs h) then
        vs
    else if k == h - 1 && vs[k] > u16.highest / 2 then
        zeroAndSet 1 0 m :> [ipb*(4*m)]u16
    else if eqBpow vs k then
        zeroAndSet 1 (h - k) m :> [ipb*(4*m)]u16
    else 
        let l = i64.min k 2
        let V = loop V = 0u64 for i < l do
            V + (u64.u16 vs[k - l + i + 1]) << 16*(u64.i64 i)
        let b2l = 1u64 << 16*2*(u64.i64 l)
        let tmp = (b2l - V) / V + 1

        let vs = tabulate (ipb*(4*m)) (\i -> 
            if i == 0 then u16.u64 tmp
            else if i == 1 then u16.u64 (tmp >> 16)
            else vs[i] )

        in if h - k <= l then
            shift (h-k-l) vs
        else
            refine us vs h k l




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

    let quo = (shinv us vs h k)
        |> bmulu16 us
        |> shift (-h)
    let rem = bmulu16 vs quo
        |> bsubu16 us

    let (quo, rem) =
        if lt rem vs then
            let quo = badd1u16 quo
            let rem = bsubu16 rem vs
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

    let quo = shinv us vs h k
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





 --let V = loop V = 0u64 for i < l do

-- ==
-- entry: test_div
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 25282u16, 2132u16, 10232u16, 8987u16, 59880u16, 52711u16, 17293u16, 3958u16, 9562u16, 63790u16, 29283u16, 49715u16, 55199u16, 50377u16, 1946u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [64358u16, 23858u16, 20493u16, 55223u16, 47665u16, 58456u16, 12451u16, 55642u16, 24869u16, 35165u16, 45317u16, 41751u16, 43096u16, 23273u16, 33886u16, 43220u16, 48555u16, 36018u16, 53453u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [19472u16, 62163u16, 27479u16, 13589u16, 47175u16, 43963u16, 55342u16, 58871u16, 55235u16, 53043u16, 2386u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [39433u16, 45455u16, 53114u16, 8163u16, 2139u16, 41117u16, 26901u16, 18168u16, 43904u16, 52648u16, 42003u16, 21686u16, 4014u16, 49277u16, 30849u16, 40590u16, 42920u16, 59996u16, 43580u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
entry test_div (us: [1*(4*16)]u16) (vs: [1*(4*16)]u16) : ([1*(4*16)]u16, [1*(4*16)]u16) =
    div us vs

-- ==
-- entry: test_quo
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 25282u16, 2132u16, 10232u16, 8987u16, 59880u16, 52711u16, 17293u16, 3958u16, 9562u16, 63790u16, 29283u16, 49715u16, 55199u16, 50377u16, 1946u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [64358u16, 23858u16, 20493u16, 55223u16, 47665u16, 58456u16, 12451u16, 55642u16, 24869u16, 35165u16, 45317u16, 41751u16, 43096u16, 23273u16, 33886u16, 43220u16, 48555u16, 36018u16, 53453u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [19472u16, 62163u16, 27479u16, 13589u16, 47175u16, 43963u16, 55342u16, 58871u16, 55235u16, 53043u16, 2386u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
entry test_quo (us: [1*(4*16)]u16) (vs: [1*(4*16)]u16) : [1*(4*16)]u16 =
    quo us vs

-- ==
-- entry: bench_div
-- compiled random input { [64]u16   [3]u16 }
-- compiled random input { [128]u16  [3]u16 }
-- compiled random input { [256]u16  [3]u16 }
-- compiled random input { [512]u16  [3]u16 }
-- compiled random input { [1024]u16 [3]u16 }
-- compiled random input { [2048]u16 [3]u16 }
-- compiled random input { [4096]u16 [3]u16 }
-- compiled random input { [8192]u16 [3]u16 }
-- compiled random input { [16384]u16 [3]u16 }
entry bench_div [m] (us: [m]u16) (vs: [3]u16) : ([]u16, []u16) =
    let vs = tabulate m (\i -> if i < 3 then vs[i] else 0)
    let ipb = 1
    let m = m / 4
    let us = (us :> [ipb*(4*m)]u16)
    let vs = (vs :> [ipb*(4*m)]u16)
    in div us vs

-- ==
-- entry: bench_quo
-- compiled random input { [64]u16   [3]u16 }
-- compiled random input { [128]u16  [3]u16 }
-- compiled random input { [256]u16  [3]u16 }
-- compiled random input { [512]u16  [3]u16 }
-- compiled random input { [1024]u16 [3]u16 }
-- compiled random input { [2048]u16 [3]u16 }
-- compiled random input { [4096]u16 [3]u16 }
-- compiled random input { [8192]u16 [3]u16 }
-- compiled random input { [16384]u16 [3]u16 }
entry bench_quo [m] (us: [m]u16) (vs: [3]u16) : []u16 =
    let vs = tabulate m (\i -> if i < 3 then vs[i] else 0)
    let ipb = 1
    let m = m / 4
    let us = (us :> [ipb*(4*m)]u16)
    let vs = (vs :> [ipb*(4*m)]u16)
    in quo us vs

def us' = [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 25282u16, 2132u16, 10232u16, 8987u16, 59880u16, 52711u16, 17293u16, 3958u16, 9562u16, 63790u16, 29283u16, 49715u16, 55199u16, 50377u16, 1946u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] :> [1*(4*16)]u16
def vs' = [64358u16, 23858u16, 20493u16, 55223u16, 47665u16, 58456u16, 12451u16, 55642u16, 24869u16, 35165u16, 45317u16, 41751u16, 43096u16, 23273u16, 33886u16, 43220u16, 48555u16, 36018u16, 53453u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] :> [1*(4*16)]u16

def x = [0,4,1,0]:> [1*(4*1)]u64
def y = [1,4,2,3] :> [1*(4*1)]u64

def foo = [0,4,1,0]:> [1*(4*1)]u16
def bar = [420, 0, 0, 0] :> [1*(4*1)]u16

def x' = [0,4,1,0,0,0,0,0]:> [1*(4*2)]u64
def y' = [1,4,2,3,0,0,0,0] :> [1*(4*2)]u64

def mads = [20, 42, 10, 4, 63, 8, 22, 1] :> [1*(4*2)]u16
def mikkel = [5, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*2)]u16


def x'' = [0,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0] :> [1*(4*4)]u64
def y'' = [1,4,2,3,0,0,0,0,0,0,0,0,0,0,0,0] :> [1*(4*4)]u64

--def x'' = [0,0,0,0,0,4,1,0]:> [1*(4*2)]u64
--def y'' = [0,0,0,0,1,4,2,3] :> [1*(4*2)]u64




--def x' = [0,4,1,0]:> [1*(4*1)]u16
--def y' = [1,4,2,3] :> [1*(4*1)]u16