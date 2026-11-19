LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY classifier IS
    PORT (
        pixel : IN unsigned(7 DOWNTO 0) := (OTHERS => '0');
        sample : IN unsigned(7 DOWNTO 0) := (OTHERS => '0');
        threshold : IN unsigned(7 DOWNTO 0) := (OTHERS => '0');
        is_match : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE rtl OF classifier IS
BEGIN
    is_match <= '1' WHEN
        ABS(to_integer(pixel) - to_integer(sample))
        <= to_integer(threshold)
        ELSE
        '0';
END ARCHITECTURE;