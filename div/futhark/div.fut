import "div-helpers"
import "big-add"
import "sqr-mul"
-- import "big-add"


--
-- Calculates (a * b) rem B^d
--
def multmod [ipb][m] (us : [ipb*(4*m)]u16) (vs : [ipb*(4*m)]u16) (d : i64) : [ipb*(4*m)]u16 = 
    let res = bmulu16 us vs
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0u16 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [n] [ipb] (us : [ipb * (4 * n)]u16) (vs : [ipb * (4 * n)]u16) (h : i64) (l : i64) : (u32, []u16) =
    let precU = prec us
    let precV = prec vs
    let L = precV + precU - l + 1

    let (sign, ret) = 
        if (precU == 0 || precV == 0) then
            let sign' = 1
            let ret' = zeroAndSet 1u16 h n
            in (sign', ret')
        else if (L >= h) then
            let ret' = bmulu16 us vs
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
def step [m][ipb] (us : [ipb*(4*m)]u16) (vs : [ipb*(4*m)]u16) (h : i64) (l : i64) (n : i64) : [ipb*(4*m)]u16 =
    let (sign, vs) = powDiff us vs h l
    let vs = bmulu16 us (vs :> [ipb*(4*m)]u16)
    let vs = shift (2 * n - h) vs
    let us = shift n us

    in if sign > 0 then
        baddu16 us vs
    else
        bsubu16 us vs

--
-- Refine the approximation of the quotient
--
def refine [m][ipb] (us : [ipb*(4*m)]u16) (vs : [ipb*(4*m)]u16) (h : u64) (k : u64) (l : u64) : [ipb*(4*m)]u16 =
    let us = shift 2 us
    
    let (_, vs, _) = loop (us, vs, l) = (us, vs, l) while h - k > l do
        let n = u64.min (h - k + 1 - l) l
        let s = u64.max 0 (k - 2 * l + 1 - 2)
        let vs = shift (-s |> i64.u64) vs
        let us = step us vs (k + l + n - s + 2 |> i64.u64) (n |> i64.u64) (l |> i64.u64)
        let us = shift (-1) us
        let l = l + n - 1
        in (us, vs, l)

    in shift (-2) vs

-- Calculates the shifted inverse
--
def shinv [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) : [n]u32 =
    if k == 0 then
        [] -- TODO: implement quo
    else if k >= h && !(eqBpow vs h) then
        vs
    else if k == h - 1 && vs[k] > u32.highest / 2 then
        zeroAndSet 1 0 m
    else if eqBpow vs k then
        zeroAndSet 1 (h - k) m
    else 

        -- TODO: Fix this shit
        let V = (u64.u64 v[k-2]) + (u64.u64 v[k-1] << (i64u64.u64 bits))
               + (u64.u64 v[k] << (i64(u64.u64 (2*bits))))
        let V = ((0 - V) / V) + 1
        let vs = tabulate m (\i -> if i <= 1 then
                                        W >> (i64(u64.u64 (bits * i)))
                                   else 0)

        in if h - k <= 2 then
            shift (h-k-2) vs
        else
            refine us vs h k 2


--
-- Implementation of multi-precision integer division using
-- the shifted inverse and classical multiplication
--
def div [n] (us : [n]u32) (vs : [n]u32) : ([n]u32, [n]u32) =
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

    in ([], [])


--
-- Implementation of multi-precision integer quotient using
-- the shifted inverse and classical multiplication
--
def quo [n] (us : [n]u32) (vs : [n]u32) : [n]u32 =
    undefined
