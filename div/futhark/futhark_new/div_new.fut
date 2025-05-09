import "div-helpers"
import "add"
import "mul"
import "sub"

--
-- Calculates (a * b) rem B^d
--
def multmod [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (d: i64) : [ipb*(4*m)]u16 = 
    let res = convMulV3 us vs
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
        let ret = convMulV3 us vs
        in if ltbpow ret h then
            let tmp = zeroAndSet 1 (h) (ipb*(4*m))
            let ret = (bsub tmp ret).0
            in (1, ret)
        else
            let bpow = zeroAndSet 1 h (ipb*(4*m))
            let (ret, _) = bsub ret bpow
            in (0, ret)
    else 
        let ret = multmod us vs L
        in if !(ez ret) && ret[L-1] == 0 then 
            (0, ret)
        else 
            let bpow = zeroAndSet 1 L (ipb*(4*m))
            let (ret, _) = bsub bpow ret
            in (1, ret)

--
-- Iterate towards an approximation in at most log(M) steps
--
def step [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (h: i64) (l: i64) (n: i64) : [ipb*(4*m)]u16 =
    let (sign, us) = (powDiff vs  us (h-n) (l-2))
    let us = convMulV3 us vs
        |> shift (2 * n - h)
    --let us = trace (shift n us)
    let vs = shift n vs
    in if sign != 0 then
        baddV3 us vs
    else
        -- bsub16 is our implementation, bsub in from the thorbjørn libary
        --let ret = trace(bsubu16 us vs)
        let ret = (bsub us vs).0
        in ret

--
-- Refine the approximation of the quotient
--
def refine [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (k: i64) (l: i64) : [ipb*(4*m)]u16 =
    let ws = (shift 2 ws)
    let (ws, _) = loop (ws, l) = (ws, l)
        while h - k > l do
            let n = i64.min (h - k + 1 - l) l
            let s = i64.max 0 (k - 2 * l + 1 - 2)
            let vs = shift (-s) vs
            let tmp = step ( vs) ( ws) (k + l + n - s + 2) n l
            let ws = shift (-1) tmp
            let l = l + n - 1
            in (ws, l)
    in shift (-2) ws


--
-- Calculates the shifted inverse
--
def shinv [m][ipb] (vs: [ipb*(4*m)]u16) (h: i64) (k: i64) : [ipb*(4*m)]u16 =
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
        --let V = loop V = 0u64 for i < l+1 do
        --    (V + (u64.u16 vs[k - l + i + 1]) << 16*(u64.i64 i))
        let V = (u64.u16 vs[k - 2]) | (u64.u16 vs[k - 1]) << 16 | (u64.u16 vs[k]) << 32 
        let b2l = 1u64 << 16*4--*(u64.i64 l)
        let tmp = (b2l - V) / (V + 1)

        let ws = tabulate (ipb*(4*m)) (\i -> 
            if i == 0 then u16.u64 tmp
            else if i == 1 then u16.u64 (tmp >> 16)
            else 0u16)
        in if h - k <= l then
            shift (h-k-l) ws
        else
            refine vs ws h k l


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

    --let quo = shinv us vs h k
    --    |> convMulV3 us
    --    |> shift (-h)
    let quo = 
        let m = m * 2
        let quo_padded = ((shinv vs h k) ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let us_padded = (us ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let mul_res = convMulV3 quo_padded us_padded
        let mul_shifted = shift (-h) mul_res
        let res = take (ipb*(4*(m/2))) mul_shifted
        in res
    let quo = quo :> [ipb * (4 * m)]u16
    --let quo = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] :> [ipb*(4*m)]u16
    let rem = (convMulV3 vs quo
        |> bsub us).0
        --|> bsubu16 us)

    let (quo, rem) =
        if not (lt rem vs) then
            let quo = badd1u16 quo
            -- same deal of different sub implementation, testing will reveal the best
            -- our bsubu16 does not work for the us' vs' input
            --let rem = bsubu16 rem vs
            let rem = ((bsub rem vs).0)
            in (quo, rem)
        else
            (quo, rem)

    let rem =
        if kIsOne then
            (shift (-1) rem)
        else 

            (rem)

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
        |> convMulV3 us
        |> shift (-h)

    let rem = convMulV3 vs quo
        |> bsubu16 us

    let quo =
        if lt rem vs then
            (badd1u16 quo)
        else
            (quo)

    in quo



-- alot of junk for testing 

-- gives correct result
def mads = [20, 42, 10, 4, 63, 8, 22, 1] :> [1*(4*2)]u16
def mikkel = [5, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*2)]u16


-- give correct result
def foo = [0,1,4,0]:> [1*(4*1)]u16
def bar = [420, 0, 0, 0] :> [1*(4*1)]u16

-- gives correct result
def cat = [4,2,2,2,0,0,0,0] :> [1*(4*2)]u16
def dog = [4,1,1,1,0,0,0,0] :> [1*(4*2)]u16

-- gives correct result
def y = [1,4,2,3] :> [1*(4*1)]u16
def x = [0,4,1,0]:> [1*(4*1)]u16

-- gives correct result
def horse = [7, 3, 5, 10, 7, 9, 7, 9, 7, 2, 2, 10, 0, 0, 0, 0] :> [1*(4*4)]u16
def donkey = [4, 4, 5, 1, 7, 2, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*4)]u16 

-- gives correct result
def us' = [39017, 18547, 56401, 23807, 37962, 22764, 7977, 31949, 0, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*4)]u16
def vs' = [22714, 55211, 16882, 7931, 43491, 57670, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*4)]u16

-- gives correct result
def knold = [4, 2, 3, 6, 9, 6, 10, 10, 10, 9, 9, 9, 0, 0, 0, 0] :> [1*(4*4)]u16 
def tot = [10, 1, 1, 5, 10, 2, 4, 4, 1, 1, 1, 0, 0, 0, 0, 0] :> [1*(4*4)]u16 

-- does not give correct result, something happens in refine, second round of powdiff gives wrong result.
def us = [35165, 45317, 41751, 43096, 23273, 33886, 43220, 48555, 36018, 53453, 57542, 0, 0, 0, 0, 0] :> [1*(4*4)]u16 
def vs = [30363, 40628, 9300, 34321, 50190, 7554, 63604, 34369, 0, 0, 0, 0, 0, 0, 0, 0] :> [1*(4*4)]u16 


--
-- ==
-- entry: test_div 
-- compiled input { [20u16, 42u16, 10u16, 4u16, 63u16, 8u16, 22u16, 1u16] [5u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [4u16, 39330u16, 39323u16, 52429u16, 13119u16, 39323u16, 13111u16, 0u16] [0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [0u16,1u16,4u16,0u16] [420u16, 0u16, 0u16, 0u16]}
-- output { [10142u16, 624u16, 0u16, 0u16] [200u16, 0u16, 0u16, 0u16] }
-- compiled input { [4u16,2u16,2u16,2u16,0u16,0u16,0u16,0u16] [4u16,1u16,1u16,1u16,0u16,0u16,0u16,0u16] }
-- output { [1u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [0u16, 1u16, 1u16, 1u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [1u16,4u16,2u16,3u16] [0u16,4u16,1u16,0u16] }
-- output { [65526u16, 2u16, 0u16, 0u16] [1u16, 44u16, 0u16, 0u16] }
-- compiled input { [7u16, 3u16, 5u16, 10u16, 7u16, 9u16, 7u16, 9u16, 7u16, 2u16, 2u16, 10u16, 0u16, 0u16, 0u16, 0u16] [4u16, 4u16, 5u16, 1u16, 7u16, 2u16, 2u16, 10u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] } 
-- output { [0u16, 0u16, 0u16, 0u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [7u16, 3u16, 5u16, 10u16, 3u16, 5u16, 2u16, 8u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [4u16, 2u16, 3u16, 6u16, 9u16, 6u16, 10u16, 10u16, 10u16, 9u16, 9u16, 9u16, 0u16, 0u16, 0u16, 0u16] [10u16, 1u16, 1u16, 5u16, 10u16, 2u16, 4u16, 4u16, 1u16, 1u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [65535u16, 8u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [14u16, 65449u16, 65530u16, 1u16, 65510u16, 65453u16, 65531u16, 65513u16, 65510u16, 0u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [54959u16, 255u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [30019u16, 15584u16, 62297u16, 30007u16, 47579u16, 3678u16, 23u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
entry test_div  [m] (us: [m]u16) (vs: [m]u16) : ([m]u16, [m]u16) =
    let mdiv4 = m / 4
    let us = us :> [1*(4*mdiv4)]u16
    let vs = vs :> [1*(4*mdiv4)]u16
    let ret = div us vs :> ([m]u16, [m]u16)
    in ret