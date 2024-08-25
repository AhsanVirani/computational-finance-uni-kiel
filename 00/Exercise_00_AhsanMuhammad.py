"""
Authored by: Ahsan Muhammad (matriculation number: 1183091)
Group Number: 12
Excercise Number: 00
"""

import math

def bond_value(V0, r, n, M, c):
    """_summary_

    Args:
        V0 (_type_): initial endowment
        r (_type_): Interest rate
        n (_type_): Number of years
        M (_type_): Number of time periods per year
        c (_type_): Flag indicating the type of interest rate (1 for continuous rate, 0 for simple rate)
    
    Returns:
        float: Capital Vn after n years
    """
    if c == 1:
        return V0 * math.exp(r * n) 
    elif c == 0: 
        return V0 * (1 + (r / M)) ** (n * M)
    
    raise ValueError("c can only take a value of 1 or 0")

def main():
    """
    Main function for testing the function bond_value and displaying value
    """
    # init vars
    V0 = 1000
    r = 0.05
    n = 10
    M = 4
    c = 0
    
    # Run the function
    b_val = bond_value(V0=V0, r=r, n=n, M=M, c=c)
    
    # Print bond value
    print(b_val)
    
if __name__ == '__main__':
    main()