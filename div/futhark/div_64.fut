import "div-helpers_64"
import "big-add"
import "sqr-mul"




--
-- Calculates (a * b) rem B^d
--
def multmod [m] [ipb] (us : [ipb*(4*m)]u64) (vs : [ipb*(4*m)]u64) (d : i64) : [ipb*(4*m)]u64 = 
    let res = bmul us vs
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0u64 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [m] [ipb] (us : [ipb * (4 * m)]u64) (vs : [ipb * (4 * m)]u64) (h : i64) (l : i64) : (u64, [ipb * (4 * m)]u64) =
    let precU = prec us
    let precV = prec vs
    let L = precV + precU - l + 1

    let (sign, ret) = 
        if (precU == 0 || precV == 0) then
            let sign' = 1
            let ret' = zeroAndSet 1u64 h (ipb * (4 * m))
            in (sign', ret')
        else if (L >= h) then
            let ret' = bmul us vs
            let sign' = 0
            in (sign', ret')
        else 
            --et sign' = 0
            let ret' = multmod us vs 2i64
            --in (sign', ret')
            in
            if (!(ez vs) ) then 
                if (vs[L-1] == 0) then 
                    let sign' = 0
                    in (sign', ret')
                else 
                    let sign' = 1
                    in (sign', ret')
            else 
                let sign' = 1
                in (sign', ret')

    in (sign,ret)

--
-- Iterate towards an approximation in at most log(M) steps
--
def step [m] [ipb] (us : [ipb * (4 * m)]u64) (vs : [ipb * (4 * m)]u64) (h : i64) (l : i64) (n : i64) : [ipb * (4 * m)]u64 =
    let (sign, vs') = powDiff us vs h l
    let vs = bmul us vs'
    let vs = shift (2 * n - h) vs
    let us = shift n us

    in if (bool.u64(sign)) then
        badd us vs
    else
        replicate (ipb * (4 * m)) 0u64 --bsub us vs

--
-- Refine the approximation of the quotient
--
def refine [m] [ipb] (us : [ipb * (4 * m)]u64) (vs : [ipb * (4 * m)]u64) (h : i64) (k : i64) (l : i64) : [ipb * (4 * m)]u64 =
    let us = shift 2i64 us
    
    let (_us, vs, _l) = loop (us, vs, l) = (us, vs, l) while h - k > l do
        let n = i64.min (h - k + 1 - l) l
        let s = i64.max 0 (k - 2 * l + 1 - 2)
        let vs = shift (-s) vs
        let us = step us vs (k + l + n - s + 2) n l
        let us = shift (-1) us
        let l = l + n - 1
        in (us, vs, l)

    in shift (-2) vs

--
-- Calculates the shifted inverse
--
def shinv [m] [ipb] (us : [ipb * (4 * m)]u64) (vs : [ipb * (4 * m)]u64) (h : i64) (k : i64) : [ipb * (4 * m)]u64 =
    if k == 0 then
        replicate (ipb * (4 * m)) 0u64 -- TODO: implement quo
    else if k >= h && !(eqBpow vs h) then
        vs
    else if k == h - 1 && vs[k] > (u64.highest / 2) then
        zeroAndSet 1 0 (ipb * (4 * m))
    else if eqBpow vs k then
        zeroAndSet 1 (h - k) (ipb * (4 * m))
    else 

        -- TODO: Fix this shit
        let V = (u64.u64 vs[k-2]) + (u64.u64 vs[k-1] << (64))
               + (u64.u64 vs[k] << (2*64))
        let V = ((0 - V) / V) + 1
        let vs = tabulate ((ipb * (4 * m))) (\i -> if i <= 1 then
                                        V >> (u64.i64(64 * i))
                                   else 0)

        in if h - k <= 2 then
            shift (h-k-2) vs
        else
            refine us vs h k 2


--
-- Implementation of multi-precision integer division using
-- the shifted inverse and classical multiplication
--
def div [m] [ipb] (us : [ipb * (4 * m)]u64) (vs : [ipb * (4 * m)]u64) : ([ipb * (4 * m)]u64, [ipb * (4 * m)]u64) =
    let h = prec us
    let k = (prec vs) - 1

    let (us, vs, h, k) =
        if k == 1 then
            let h = h + 1
            let k = k + 1
            let us = shift 1 us
            let vs = shift 1 vs
            in (us, vs, h, k)
        else
            (us, vs, h, k)


    -- let tmp = shinv us vs h k
    -- let tmp = bmul us tmp
    -- let tmp = shift (-h) tmp
    -- let tmp = bmul us tmp

    -- TODO: implement sub

    -- in if lt tmp vs then
        -- TODO: implement add1
        -- TODO: implement sub
        -- if 

    -- in if k == 1 then
    --     shift (-1) tmp
    -- else 
    --     tmp

    in (replicate (ipb * (4 * m)) 0u64, replicate (ipb * (4 * m)) 0u64)



--
-- Implementation of multi-precision integer quotient using
-- the shifted inverse and classical multiplication
--
--def quo [n] (us : [n]u32) (vs : [n]u32) : [n]u32 =
--    undefined
