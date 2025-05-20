type cT         = u32      --u8
let  cTfromBool = u32.bool --u8.bool
let  two_cT     = 2u32     --2u8

let carryOpNE: cT = two_cT

let carryOp (c1: cT) (c2: cT) =
  (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1)
  
let carrySegOp (c1: cT) (c2: cT) =
    if (c2 & 4) != 0 then c2
    else let res = ( (c1 & (c2 >> 1)) | c2 ) & 1
         let res = res | (c1 & c2  & 2)
         in  ( res | ( (c1 | c2) & 4 ) )

let badd1u16 [ipb][m] (us : [ipb*(4*m)]u16) : [ipb*(4*m)]u16 =
  let min_idx = reduce u16.min u16.highest us |> i64.u16
  in tabulate (ipb* (4*m)) (\i -> 
    if (i == min_idx) then us[i] + 1
    else us[i] )

-- check the precision of bigint, eg n - (leading zero's)
def prec [n] (u : [n]u16) : (i64) = 
    let bar = reduce (\ idx1 idx2 -> 
                        if u[idx2] != 0 then (i64.max idx2 idx1) else idx1) (0i64) (iota n)
    let bar = if (bar == 0) && (u[0]== 0 ) then 0 else bar + 1
    in bar

-- checks if two bigints are equal
def eq [n] (u : [n]u16) (v : [n]u16) : bool =
    reduce (\ x y -> (x == y && x != false)) true (map2 (==) u v)


-- cheaks if given bigint = 0
def ez [n] (u : [n]u16) : bool = 
    all (== 0) u    


-- less than taken from thorbjørn, musch cleaner than mine 
-- source : https://github.com/tossenxD/big-int/blob/main/futhark/helper.fut
def lt [m][ipb] (u: [ipb*(4*m)]u16) (v: [ipb*(4*m)]u16) : bool =
  let res = map2 (\ x y -> (x < y, x == y) ) u v
  |> reduce (\ (l1, e1) (l2, e2) -> (l2 || (e2 && l1), e1 && e2) ) (false, true)
  in res.0



-- set a given index of the bigint to 
def set [n] (u : *[n]u16) (d : u16) (idx : i32) : [n]u16 = 
    let u[idx] =  d in u

-- zero bigint array and set given index to d
def zeroAndSet (d : u16) (idx : i64) (m : i64) : [m]u16 = 
    tabulate m (\i -> if i == idx then d else 0u16)

def zeroAndSet_inplace [n] (d : u32) (idx : i64) (arr : *[n]u32) : []u32 = 
    let arr[idx] = d
    in arr
    
def ltbpow [m][ipb] (u: [ipb*(4*m)]u16) (bpow: i64) : bool =
  let v = zeroAndSet 1 bpow (4 * m) :> [ipb*(4*m)]u16
  let res = map2 (\ x y -> (x < y, x == y) ) u v
  |> reduce (\ (l1, e1) (l2, e2) -> (l2 || (e2 && l1), e1 && e2) ) (false, true)
  in res.0
 
def shift [m][ipb] (shft : i64) (u : [ipb*(4*m)]u16) : ([ipb*(4*m)]u16) =
    map (\ idx -> let offset = idx - shft 
           in if offset < (ipb*(4*m)) && offset >= 0 then u[offset] else 0) (iota (ipb*(4*m)))

def eqBpow [m][ipb] (u : [ipb*(4*m)]u16) (b : i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m))
    in reduce (\ x y -> (x == y && x != false)) true (map2 (==) u bpow)

def ltBpow [m][ipb] (u: [ipb*(4*m)]u16) (b: i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m))
    in lt u bpow

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



let subbpowbigint [ipb][m] (bpow : i64) (us : [ipb*(4*m)]u16) : [ipb*(4*m)]u16 =
  let min_idx = (reduce u16.min (trace u16.highest) (trace us)) |> i64.u16
  let min_idx = trace min_idx
  in tabulate (ipb* (4*m)) (\i -> 
    if (i < min_idx) then 0
    else if i < bpow then (if i == min_idx then (! us[i]) + 1 else (! us[i]))   --1 - us[i]
    else us[i] )

let subbigintbpow [ipb][m] (us : [ipb*(4*m)]u16) (bpow : i64) : [ipb*(4*m)]u16 =
  let min_idx = reduce u16.min u16.highest us |> i64.u16
  in tabulate (ipb* (4*m)) (\i -> 
    if (i >= bpow && i <= min_idx) then us[i] - 1
    else us[i] )

let subPairwiseu16 (m: i32) (ash: []u16) (bsh: []u16) (tid: i32) (i: i32) : (u16, cT)=
  let ind = tid * 4 + i
  let (a,b) = ( #[unsafe] ash[ind], #[unsafe] bsh[ind] )
  let r = a - b
  let c = cTfromBool (r > a)
  let c = c | ( (cTfromBool (r == 0u16)) << 1 )
  let c = c | ( (cTfromBool ( (ind % m) == 0 )) << 2 )
  in  (r, c)

let bsub0u16 (ipb: i64) (n:i64) (ash : []u16) (bsh : []u16) : [ipb*(4*n)]u16 =
     let nn = i32.i64 n
  let g = ipb * n
  let seqred4 (tid: i32) =
    loop (accum) = (carryOpNE) for i < 4 do
        let (_, c) = subPairwiseu16 (4 * nn) ash bsh tid i
        in  carrySegOp accum c
  
  let seqscan1 (tid: i32) (i: i32) (carry: cT) =
    let (r0, c0) = subPairwiseu16 (4 * nn) ash bsh tid i
    let r0 = r0 - u16.bool ( ( (c0 & 4) == 0 ) && ( (carry & 1) == 1 ) ) 
    in  (r0, carrySegOp carry c0)

  let seqscan4 (carries: [g]cT) (tid: i32) =
    let carry = if tid == 0 then carryOpNE else #[unsafe] carries[tid-1] 
    let (r0, carry) = seqscan1 tid 0 carry
    let (r1, carry) = seqscan1 tid 1 carry
    let (r2, carry) = seqscan1 tid 2 carry
    let (r3, _)     = seqscan1 tid 3 carry
    in  (r0,r1,r2,r3)  

  let carries = iota g
             |> map i32.i64
             |> map seqred4
             |> scan carrySegOp carryOpNE 

  let (rs0, rs1, rs2, rs3) = iota g |> map i32.i64 
                          |> map (seqscan4 carries)
                          |> unzip4
  let rs = rs0 ++ rs1 ++ rs2 ++ rs3 
  let rs = (rs :> [ipb*(4*n)]u16) |> opaque
  in  rs




-- our implementation of big-int subtraction.
let bsubu16 [ipb][n] (as : [ipb*(4*n)]u16) (bs : [ipb*(4*n)]u16) : [ipb*(4*n)]u16 =
  let g = ipb * n

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
  let ash = ash |> opaque
  let bsh = bsh |> opaque
  
  in  (bsub0u16 ipb n ash bsh) :> [ipb*(4*n)]u16






entry test_quo_single [m] (bpow : i64) (d :[m]u16) (n : i64) : ([m]u16) =
    let mdiv4 = m / 4
    let ds = d :> [1 * (4 * mdiv4)]u16
    let ret = (quo_single bpow ds n) :> [m]u16
    in ret

entry test_lt [m] (us: [m]u16) (vs: [m]u16) : bool =
    let mdiv4 = m / 4
    let us' = us :> [1 * (4 * mdiv4)]u16
    let vs' = vs :> [1 * (4 * mdiv4)]u16
    in
    lt us' vs'

entry test_prec [n] (u : [n]u16) : (i64) = 
    prec u 

