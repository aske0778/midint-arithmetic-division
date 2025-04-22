import "div-helpers"
import "sqr-mul"
-- import "big-add"


--
-- Calculates (a * b) rem B^d
--
def multmod [n] (us : [n]u32) (vs : [n]u32) (d : u32) : ([n]u32) = 
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

    if (precU == 0 || precV == 0) then
        let retval = zeroAndSet vs 1 h
        in (sign, retval)
    else if (L >= h) then
        let res = bmul us vs
        if (ltBpow vs h) then
            (1, iota m) -- TODO: fix
            -- (1, sub)
        else
            (0, iota m) -- TODO: fix
            -- (0, sub)
    else 
        let mlt = multmod us vs L
        if (!ez vs) then
            (0, iota m)
        


--
-- Iterate towards an approximation in at most log(M) steps
--
def step [n] (us : [n]u32) (vs : [n]u32) (h : u32) (l : u32) (n : u32) : ([n]u32) =
    undefined

def refine [n] (us : [n]u32) (vs : [n]u32) (h : u32) (l : u32) (k : u32) : ([n]u32) =
    undefined

def shinV [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) : ([n]u32) =
    undefined

def divShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32, [n]u32) =
    undefined

def quoShinV [n] (us : [n]u32) (vs : [n]u32) : ([n]u32) =
    undefined
