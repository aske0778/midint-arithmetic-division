
-- check the precision of bigint, eg n - (leading zero's)
def prec [n] (u : [n]u16) : (i64) = 
    let bar = reduce (\idx1 idx2 -> 
                        if u[idx2] != 0 then (idx2 + 1) else idx1) (0i64) (iota n)
    in bar

-- checks if two bigints are equal
def eq [n] (u : [n]u16) (v : [n]u16) : bool =
    reduce (\ x y -> (x == y && x != false)) true (map2 (==) u v)


-- cheaks if given bigint = 0
def ez [n] (u : [n]u16) : bool = 
    all (== 0) u    

-- returns u < v for two bigint
--def lt [n] (u : [n]u32) (v : [n]u32) : bool = 
--    let temp =  (map2 (\ x y -> i32.u32(x) - i32.u32(y)) (reverse u) (reverse v))
--    let temp2 =  (reduce (\ne x -> if (((x) > 0 && (ne) == 1) || ne == 0) then 0 
--                                        else if  (x == 0 && ne != -1) then 1 
--                                        else if  (x < 0 && ne == 1) then -1
--                                        else -1) 1 temp)
--    in 
--        if temp2 == -1 then true else false

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
  let v = zeroAndSet 1 bpow m :> [ipb*(4*m)]u16
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

-- Testing ez
-- ==
-- entry: ez_test 
-- compiled input { [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32] }
-- output { true }
-- compiled input { [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32] }
-- output { false }
-- compiled input { [0u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32] }
-- output { false }
-- compiled input { [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 1u32] }
-- output { false }

-- Testing lt 
-- == 
-- entry: lt_test
-- compiled input {   [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 1u32]}
-- output { true }

-- Testing eq
-- == 
-- entry: eq_test
-- compiled input {   [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [1u32, 0u32, 1u32, 0u32, 1u32, 0u32, 0u32, 1u32]
--                    [1u32, 0u32, 1u32, 0u32, 1u32, 0u32, 0u32, 1u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 1u32, 0u32, 1u32, 0u32, 1u32]
--                    [2u32, 0u32, 1u32, 1u32, 0u32, 1u32, 0u32, 1u32]}
-- output { true }


--                    [2u32,  0u32,  1u32,  0u32, 0u32,  1u32,  0u32,  0u32]
--                    [1u32,  0u32,  0u32,  1u32, 0u32,  0u32,  0u32,  1u32]
--                    [false, false, false, true, false, false, false, true]

--                    [true, false, false, false, true, false, false, false]

--                    [2u32,  0u32,  1u32,  0u32, 0u32,  1u32,  1u32,  1u32]
--                    [1u32,  0u32,  0u32,  1u32, 0u32,  0u32,  2u32,  1u32]
--                    [false, false, false, true, false, false, true, false]