import "div-helpers"
import "sqr-mul"
-- import "big-add"




--
-- Calculates (a * b) rem B^d
--
def multmod [ipb][m] (us : [ipb*(4*m)]u32) (vs : [ipb*(4*m)]u32) (d : u32) : [ipb*(4*m)]u32 = 
    let res = bmul (map u64.u32 us) (map u64.u32 vs)
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [n] [ipb] (us : [ipb * (4 * n)]u32) (vs : [ipb * (4 * n)]u32) (h : u32) (l : u32) : (u32, []u32) =
    let precU = prec us
    let precV = prec vs
    let L = precV + precU - l + 1

    let (sign, ret) = 
        if (precU == 0 || precV == 0) then
            let sign' = 1
            let ret' = zeroAndSet h 1i64 n
            in (sign', ret')
        else if (L >= h) then
            let ret' = bmul us vs
            let sign' = 0
            in (sign', ret')
        else 
            --let sign' = 0
            let ret' = multmod us vs L
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

    --let sign = 1
    --in
    --if (precU == 0 || precV == 0) then
    --    let retval = zeroAndSet vs 1 h
    --    in (sign, retval)
    --else if (L >= h) then
    --    let res = bmul us vs
    --    in
    --    if (ltBpow vs h) then
    --        (1, iota m) -- TODO: fix
    --        -- (1, sub)
    --    else
    --        (0, iota m) -- TODO: fix
    --        -- (0, sub)
    --else 
    --    let mlt = multmod us vs L
    --    in
    --    if (!(ez vs)) then
    --        if (vs[L-1] == 0) then 
    --            sign <- 0
    --            (0, iota m)
    --        else 
    --            sign <- 42069
    --            (0, iota m)
    --        --(--0, iota m)
    --in sign
def powDiff [n] [ipb] (us : [ipb * (4 * n)]u32) (vs : [ipb * (4 * n)]u32) (h : u32) (l : u32) : (u32, []u32) =
    let precU = prec us
    let precV = prec vs
    let L = precV + precU - l + 1

    let (sign, ret) = 
        if (precU == 0 || precV == 0) then
            let sign' = 1
            let ret' = zeroAndSet h 1i64 n
            in (sign', ret')
        else if (L >= h) then
            let ret' = bmul us vs
            let sign' = 0
            in (sign', ret')
        else 
            --let sign' = 0
            let ret' = multmod us vs L
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

    --let sign = 1
    --in
    --if (precU == 0 || precV == 0) then
    --    let retval = zeroAndSet vs 1 h
    --    in (sign, retval)
    --else if (L >= h) then
    --    let res = bmul us vs
    --    in
    --    if (ltBpow vs h) then
    --        (1, iota m) -- TODO: fix
    --        -- (1, sub)
    --    else
    --        (0, iota m) -- TODO: fix
    --        -- (0, sub)
    --else 
    --    let mlt = multmod us vs L
    --    in
    --    if (!(ez vs)) then
    --        if (vs[L-1] == 0) then 
    --            sign <- 0
    --            (0, iota m)
    --        else 
    --            sign <- 42069
    --            (0, iota m)
    --        --(--0, iota m)
    --in sign
        


--
-- Iterate towards an approximation in at most log(M) steps
--
def step [n] (us : [n]u32) (vs : [n]u32) (h : i32) (l : i32) (n : i32) : [n]u32 =
    let (sign, vs) = powDiff us vs h l
    let vs = bmul us vs
    let vs = shift (2 * n - h) vs
    let us = shift n us

    in if sign then
        badd us vs
    else
        bsub us vs

--
-- Refine the approximation of the quotient
--
def refine [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) (l : u32) : [n]u32 =
    let us = shift 2 us
    
    let (us, vs, l) = loop (us, vs, l) = (us, vs, l) while h - k > l do
        let n = min (h - k + 1 - l) l
        let s = max 0 (k - 2 * l + 1 - 2)
        let vs = shift (-s) vs
        let us = step us vs (k + l + n - s + 2) n l
        let us = shift (-1) us
        let l = l + n - 1
        in (us, vs, l)

    in shift (-2) vs

--
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
