import "div-helpers"
import "sqr-mul"
-- import "big-add"




--
-- Calculates (a * b) rem B^d
--
def multmod [n] (us : [n]u32) (vs : [n]u32) (d : u32) : [n]u32 = 
    let res = bmul us vs
    in tabulate n (\i -> if i >= d then 0 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [n] (us : [n]u32) (vs : [n]u32) (h : u32) (l : u32) : (u32, []u32) =
    let precU = prec u
    let precV = prec v
    let L = precV + prevU - l + 1
    let sign = 1

    in if (precU == 0 || precV == 0) then
        let retval = zeroAndSet vs 1 h
        in (sign, retval)
    else if (L >= h) then
        let res = bmul us vs
        in if (ltBpow vs h) then
            (1, iota m) -- TODO: fix
            -- (1, sub)
        else
            (0, iota m) -- TODO: fix
            -- (0, sub)
    else 
        let mlt = multmod us vs L
        in if (!ez vs) then
            if (ez vs[L-1]) then
                (0, vs)
            else
                (sign, mlt) -- TODO: implement sub
        else
            (sign, mlt)
        

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
        let V = 0u64

        -- TODO: figure out this shit

        in if h - k <= 2 then
            shift (h-k-2) vs
        else
            refine us vs h k 2
        



--
-- Implementation of multi-precision integer division using
-- the shifted inverse and classical multiplication
--
def divShinv [n] (us : [n]u32) (vs : [n]u32) : ([n]u32, [n]u32) =
    undefined

--
-- Implementation of multi-precision integer quotient using
-- the shifted inverse and classical multiplication
--
def quoShinv [n] (us : [n]u32) (vs : [n]u32) : [n]u32 =
    undefined
