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
def step [m] (us : [m]u32) (vs : [m]u32) (h : i32) (l : i32) (n : i32) : [m]u32 =
    let (sign, vs) = powDiff us vs h l
    let vs = bmul us vs
    let vs = shift (2 * n - h) vs
    let us = shift n us

    in if sign then
        baddu16 us vs
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
--def shinV [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) : ([n]u32) =
--    undefined
--
--def divShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32, [n]u32) =
--    undefined
--
--def quoShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32) =
--    undefined
