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
def refine [n] (us : [n]u32) (vs : [n]u32) (h : u32) (k : u32) (l : u32) : [n]u32 =
    let us = shift 2 us
    
    let (us, vs, l) = loop (us, vs, l) = (us, vs, l) while h - k > l do
        let n = min (h - k + 1 - l) l   
        let s = max 0 (k - 2 * l + 1 - 2)
        let vs = shift (-s) vs
        let us = step us vs (k +

  let cp2sh (i : i32) = #[unsafe]
        let g = i32.i64 g in
        ( ( as[i], as[g + i], as[2*g + i], as[3*g + i] )
        , ( bs[i], bs[g + i], bs[2*g + i], bs[3*g + i] ) )

  let ( ass, bss ) = iota g |> map i32.i64
                  |> map cp2sh  |> unzip
  let (a1s, a2s, a3s, a4s) = unzip4 ass
  let (b1s, b2s, b3s, b4s) = unzip4 bss
  let ash = a1s ++ a2s ++ a3s ++ a4s
  let bsh = b1s ++ b2s ++ b3s ++ b4s
  let ash = ash |> opaque |> map u64.u16
  let bsh = bsh |> opaque |> map u64.u16
  
  in  (badd0 ipb n ash bsh) :> [ipb*(4*n)]u64 |> map u16.u64 l + n - s + 2) n l
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
