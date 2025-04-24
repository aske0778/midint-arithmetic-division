import "div-helpers-u16"
import "sqr-mul"
-- import "big-add"


--
-- Calculates (a * b) rem B^d
--
def multmod [n] [ipb] (us : [ipb * (4 * n)]u16) (vs : [ipb * (4 * n)]u16) (d : i64) : ([]u16) = 
    let res = bmulU16 us vs
    in tabulate (ipb * (4 * n)) (\i -> if i >= d then 0u16 else res[i])

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
            let ret' = bmulU16 us vs
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
--def step [n] (us : [n]u32) (vs : [n]u32) (h : u32) (l : u32) (s : u32) : ([n]u32) =
--    undefined
--
--def refine [n] (us : [n]u32) (vs : [n]u32) (h : u32) (l : u32) (k : u32) : ([n]u32) =
--    undefined
--
--def shinV [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) : ([n]u32) =
--    undefined
--
--def divShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32, [n]u32) =
--    undefined
--
--def quoShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32) =
--    undefined
