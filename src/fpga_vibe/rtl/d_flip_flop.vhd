LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY d_flip_flop IS
    PORT (
        clk : IN STD_LOGIC := '0';
        rst : IN STD_LOGIC := '0';
        d : IN STD_LOGIC := '0';
        q : OUT STD_LOGIC := '0'
    );
END ENTITY;

ARCHITECTURE rtl OF d_flip_flop IS
BEGIN
    main : PROCESS (clk, rst)
    BEGIN
        IF rising_edge(clk) THEN
            q <= d;
        ELSE
            IF rising_edge(clk) THEN
                q <= d;
            END IF;
        END PROCESS main;
    END ARCHITECTURE;