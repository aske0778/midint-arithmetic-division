

def prec [n] (u : [n]u32) : (u32) = 
    let bar = reduce (\idx1 idx2 -> 
                        if u[idx2] != 0 then idx2 else idx1) (0i64) (iota n)
    in u32.i64(bar)

-- cheaks if given bigint = 0
def bigint_ez [n] (u : [n]u32) : bool = 
    all (== 0) u 

-- returns u < v for two bigint
def bigint_lt [n] (u : [n]u32) (v : [n]u32) : bool = 
    let temp =  (map2 (\ x y -> i32.u32(x) - i32.u32(y)) (reverse u) (reverse v))
    let temp2 =  (reduce (\ne x -> if (((x) > 0 && (ne) == 1) || ne == 0) then 0 
                                        else if  (x == 0 && ne != -1) then 1 
                                        else if  (x < 0 && ne == 1) then -1
                                        else -1) 1 temp)
    in 
        if temp2 == -1 then true else false

-- checks if two bigints are equal
def bigint_eq [n] (u : [n]u32) (v : [n]u32) : bool =
    reduce (\ x y -> (x == y && x != false)) true (trace (map2 (==) u v))

def set [n] (u : *[n]u32) (d : u32) (idx : i64) : [n]u32 = 
    let u[idx] =  d in u
    
def set_and_zero (d : u32) (m : i64) : ([]u32) = 
    let foo = replicate m 0u32 
    let foo[0] = d
    in foo

def shift [n] (shft : i64) (u : [n]u32) : ([n]u32) = 
    let foo = replicate shft 0u32
    let bar = if n > 0 then (foo ++ u[:(n - shft)]) :> [n]u32 
                       else (u[(n - shft):] ++ foo) :> [n]u32
    in bar




entry ez_test = bigint_ez

entry lt_test = bigint_lt

entry eq_test = bigint_eq

-- Testing bigint_ez
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

-- Testing bigint_lt 
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

-- Testing bigint_eq
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