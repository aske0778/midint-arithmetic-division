
-- check the precision of bigint, eg n - (leading zero's)
def prec [n] (u : [n]u16) : (i64) = 
    let bar = reduce (\idx1 idx2 -> 
                        if u[idx2] != 0 then idx2 else idx1) (0i64) (iota n)
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
 

-- performs shift operation
--def shift [n] (shft : i64) (u : [n]u32) : ([n]u32) = 
--    let foo = replicate shft 0u32
--    let bar = if n > 0 then (foo ++ u[:(n - shft)]) :> [n]u32 
--                       else (u[(n - shft):] ++ foo) :> [n]u32
--    in bar

-- performs shift operation. without the use of concatination 
-- concatination is often very memory expensive. 
-- less than taken from thorbjørn, musch cleaner than mine 
-- source : https://github.com/tossenxD/big-int/blob/main/futhark/helper.fut
def shift [m][ipb] (shft : i64) (u : [ipb*(4*m)]u16) : ([ipb*(4*m)]u16) =
    map (\ idx -> let offset = idx - shft 
           in if offset < (ipb*(4*m)) && offset >= 0 then u[offset] else 0) (iota (ipb*(4*m)))

-- performs shift operation on bigint of size 2m
-- do we need a double version in futhark ?
--def shiftDouble [n] (shft : i64) (u : [n]u32) : ([n]u32) = 
--    undefined

-- Quotient calculation of a bpow and divisor d
--def quo (bpow : u32) (d : u32) : ([]u32) = 
--    let r = 1i64

def eqBpow [m][ipb] (u : [ipb*(4*m)]u16) (b : i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m))
    in reduce (\ x y -> (x == y && x != false)) true (map2 (==) u bpow)

def ltBpow [m][ipb] (u: [ipb*(4*m)]u16) (b: i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m))
    in lt u bpow


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