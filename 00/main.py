import math

def bond_value(V0, r, n, M, c):
    """_summary_

    Args:
        V0 (_type_): initial endowment
        r (_type_): interest rate
        n (_type_): _description_
        M (_type_): _description_
        c (_type_): r refers to continuous rate if c=1 elif 
            c=0 then a simple rate paid over M time periods per year
    """
    if c == 1:
        return V0*math.exp(r*n) 
    elif c == 0: 
        return V0*pow((1+(r/M)), (n*M))
    
    raise ValueError("c can only take a value of 1 or 0")

if __name__ == '__main__':
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