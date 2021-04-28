import cplex

def getOptimalAssortment(n,v,r,C,log = True):
    
    p = cplex.Cplex()
    if not log:
        p.set_log_stream(None)
        p.set_error_stream(None)
        p.set_warning_stream(None)
        p.set_results_stream(None)
        
    obj = [0]+r
    p.objective.set_sense(p.objective.sense.maximize)
    p.variables.add(obj = obj,names = ['w'+str(i) for i in range(n+1)])
    rows = []
    rowP = [1 for i in range(n+1)]
    rowC = [-C]+[1.0/v[i] for i in range(n)]
    va = [i for i in range(n+1)]
    rows = [[va,rowP],[va,rowC]]
    for i in range(1,n+1):
        temp = [0 for j in range(n+1)]
        temp[i] = 1/v[i-1] 
        temp[0] = -1
        rows.append([va,temp])
    rhs = [0 for i in range(n+2)]
    rhs[0] = 1
    p.linear_constraints.add(lin_expr=rows,senses="L"*(n+2),rhs = rhs)
    p.write("lpex.lp")
    p.solve()
    val = p.solution.get_objective_value()
    re = p.solution.get_values()
    
    #print(re)
    #print(val)
    
    x = []
    for i in range(1,n+1):
        if re[i]>0: x.append(i)
    return x
    

if __name__=='__main__':
    n = 5
    v = [1 for i in range(1,n+1)]
    r = [i for i in range(1,n+1)]
    
    print(getOptimalAssortment(n,v,r,5,log = False))
    
    
    
    