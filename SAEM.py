#######################################################################
#
# SAEM.py - semi-analytical element model
#
# a groundwater flow model for multilayer, transient systems
# based on superposition of numerical transient line source models
#
# by Walt McNab (2020)
#
#######################################################################

import sys
from numpy import *
from pandas import *
from scipy import interpolate
from scipy.sparse import *
from scipy.sparse.linalg import *
import matplotlib.pyplot as plt


### model classes ###


class Grid:
    
    def __init__(self):
        
        # read grid attributes (space and time)
        inFile = open('grid.txt','r')
        lineInput = []
        for line in inFile:
            lineInput.append(line.split())
        inFile.close()    
        self.x0 = float(lineInput[0][1])
        xf = float(lineInput[1][1])
        self.nx = int(lineInput[2][1])         
        self.y0 = float(lineInput[3][1])
        yf = float(lineInput[4][1])                
        self.ny = int(lineInput[5][1])
        self.tOut = array(lineInput[6][1:]).astype(float)
        tMonSlice = float(lineInput[7][1])
        numMonSlices = int(lineInput[8][1])

        # create uniform grid
        self.dx = (xf-self.x0)/self.nx
        self.dy = (yf-self.y0)/self.ny
        xGrid = arange(self.x0, xf, self.dx) + 0.5*self.dx
        yGrid = arange(self.y0, yf, self.dy) + 0.5*self.dy
        X, Y = meshgrid(xGrid,yGrid)
        self.x = X.flatten()
        self.y = Y.flatten()

        # time slices for monitor points        
        self.tMonOut = arange(tMonSlice, tMonSlice*numMonSlices + 0.5*tMonSlice, tMonSlice)
        print('Read and processed grid attributes.')


class Connection:           # cell-to-cell connections
    
    def __init__(self, model, aquifer):
        
        self.node0 = []     # connection node indices
        self.node1 = []
        self.d0 = []        # distances from connecting nodes to interface
        self.d1 = []
        self.KMean = []     # weighted harmonic mean hydraulic conductivity across connection
        self.rInterface = []    # connection geometry attributes for horizontal unconfined flow        
        self.bottom = []
        self.top = []
        self.b = []
        self.bottomCellBase = []    # connection geometry attributes for vertical unconfined flow
        self.topCellBase = []        
        self.insideFace = []
        self.outsideFace = []
        self.delta = 0.0               # distance into outer boundary node (for calculating gradient)

        # horizontal grid attributes
        self.rFace = self.Gridder(model)                                 # array of grid cell interface radii
        self.rNode = 0.5*self.rFace[1:] + 0.5*self.rFace[:-1]           # radius of node point associated with each cell
        self.rNode = insert(self.rNode, 0, 0.)                          # add cell representing well
       
        # connection indices arrays and supporting parameters
        self.numHconnects = model.numLayers * (model.numCols-1) + model.numLayers
        for i in range(model.numLayers):          # horizontal connections
            self.rInterface.extend(list(self.rFace))       
            for j in range(model.numCols-1):
                node0 = j + i*model.numCols
                self.node0.append(node0)
                self.node1.append(node0 + 1)
                self.bottom.append(aquifer.bottom[i])
                self.top.append(aquifer.top[i])                
                self.b.append(aquifer.b[i])
                interiorDelta = (self.rFace[j] - self.rNode[j])*(j>0) + (j==0)*model.wellDist
                self.d0.append(interiorDelta)
                self.d1.append(self.rNode[j+1] - self.rFace[j])
                self.KMean.append(aquifer.Kh[i])
                
            # horizontal boundary connections
            self.node0.append(model.numCols-1 + i*model.numCols)
            self.node1.append(model.numNodes - 1)
            self.bottom.append(aquifer.bottom[i])
            self.top.append(aquifer.top[i])                
            self.b.append(aquifer.b[i])
            self.d0.append(self.rFace.max() - self.rNode.max())
            self.d1.append(self.delta)
            self.KMean.append(aquifer.Kh[i])        # connnection K = interior node's K

        for i in range(model.numCols):                # vertical connections
            for j in range(model.numLayers-1):
                node0 = j*model.numCols + i
                self.node0.append(node0)
                self.node1.append(node0 + model.numCols)
                self.bottomCellBase.append(aquifer.bottom[j])
                d0 = 0.5*aquifer.b[j]
                d1 = 0.5*aquifer.b[j+1]
                self.d0.append(d0)                 
                self.d1.append(d1)
                self.KMean.append(HarMeanWt(aquifer.Kv[j], aquifer.Kv[j+1], d0, d1))
                self.topCellBase.append(aquifer.top[j])
                if i == 0: self.insideFace.append(0.)
                else: self.insideFace.append(self.rFace[i-1])
                self.outsideFace.append(self.rFace[i])

        # convert lists to arrays
        self.node0 = array(self.node0)
        self.node1 = array(self.node1)
        self.d0 = array(self.d0)
        self.d1 = array(self.d1)
        self.KMean = array(self.KMean)
        self.rInterface = array(self.rInterface)
        self.bottom = array(self.bottom)
        self.top = array(self.top)
        self.b = array(self.b)
        self.bottomCellBase = array(self.bottomCellBase)
        self.topCellBase = array(self.topCellBase)        
        self.insideFace = array(self.insideFace)
        self.outsideFace = array(self.outsideFace)
        print('Set up numerical model grid cell connections.')
        
    def Gridder(self, model):
        # generate radial grid
        index = arange(0, model.numCols, 1)
        f = 10.**(log10((model.boundaryRadius/model.wellRadius))/(model.numCols-1))   # sequential scaling factor
        r = model.wellRadius * f**index
        return r        
        
    def FindArea(self, h):
        # find saturated interfacial areas across cell interfaces
        hConnect = 0.5*h[self.node0[:self.numHconnects]] + 0.5*h[self.node1[:self.numHconnects]]
        dz = self.b*(hConnect>=self.top) + hConnect*((hConnect>self.bottom) & (hConnect<self.top))\
            + 0.*(hConnect<=self.bottom)
        aHoriz = 2. * pi * self.rInterface[:self.numHconnects] * dz
        vConnect = (h[self.node0[self.numHconnects:]] > self.bottomCellBase) \
            * (h[self.node1[self.numHconnects:]] > self.topCellBase)
        aVert = vConnect * pi* (self.outsideFace**2 - self.insideFace**2) 
        return concatenate((aHoriz, aVert))

    def WriteConnects(self, h):
        # convert to data frame and write to file
        area = self.FindArea(h)
        output = DataFrame(data={'node0':self.node0, 'node1':self.node1, 'd0':self.d0,
            'd1':self.d1, 'KMean':self.KMean, 'area':area})
        output.to_csv('connections.csv', index=False)
        print('Wrote connection information to file.')


class Node:         # volume element properties
    
    def __init__(self, model, aquifer, connection):
        self.index = []
        self.r = []
        self.layer = []
        self.Ss = []
        self.Sy = []
        self.top = []
        self.baseA = []
        self.vol = []
        for i in range(model.numLayers):
            self.r.extend(list(connection.rNode))
            for j in range(model.numCols):
                self.index.append(j + i*model.numCols)
                self.layer.append(i)
                self.Ss.append(aquifer.Ss[i])
                self.Sy.append(aquifer.Sy[i])
                self.top.append(aquifer.top[i])
                if j == 0: baseA = pi * connection.rFace[j]**2
                else: baseA = pi * (connection.rFace[j]**2 - connection.rFace[j-1]**2)
                self.baseA.append(baseA)
                self.vol.append(baseA * aquifer.b[i])
        # boundary node
        self.index.append(model.numNodes-1)
        self.r.append(connection.rFace.max()+connection.delta)
        self.layer.append(-1)
        self.Ss.append(aquifer.Ss.mean())
        self.Sy.append(aquifer.Sy.mean())
        self.top.append(aquifer.top.max())
        self.baseA.append(1e+30)
        self.vol.append(1e+30)
        # convert lists to arrays
        self.index = array(self.index)
        self.r = array(self.r)
        self.layer = array(self.layer)
        self.Ss = array(self.Ss)
        self.Sy = array(self.Sy)
        self.top = array(self.top)
        self.baseA = array(self.baseA)
        self.vol = array(self.vol)
        print('Set up numerical model cells.')

    def FindStorage(self, h):
        # select specific storage or specific yield, as appropriate
        S = self.Ss*self.vol*(h>=self.top) + self.Sy*self.baseA*(h<self.top)
        return S

    def WriteNodes(self, h):
        # convert to data frame and write to file
        S = self.FindStorage(h)
        output = DataFrame(data={'node':self.index, 'radius':self.r, 'layer':self.layer,
            'top':self.top, 'base area':self.baseA, 'volume':self.vol, 'storage':S})
        output.to_csv('nodes.csv', index=False)
        print('Wrote grid cell information to file.')


class Model:        # numerical multilayer radial flow model
    
    def __init__(self):
        
        # read constraints from file
        lineInput = []
        inputFile = open('params.txt','r')
        for line in inputFile:
            lineInput.append(line.split()[1])
        inputFile.close()
        self.numLayers = int(lineInput[0])
        self.numCols = int(lineInput[1])
        self.wellRadius = float(lineInput[2])
        self.boundaryRadius =float(lineInput[3])        # set boundary very far away to avoid effects
        self.h0 = float(lineInput[4])
        self.dhMax = float(lineInput[5])                # maximum head change per time step
        self.dtMin = float(lineInput[6])
        self.dtMax = float(lineInput[7])       
        self.dtPlus = float(lineInput[8])               # changes to time step (growth or reduction)
        self.dtMinus = float(lineInput[9])
        self.tEnd = float(lineInput[10])                # end-of-simulation time
        self.wellDist = float(lineInput[11])            # distance from wellbore node to well screen (a "knob")
        self.numNodes = self.numLayers * self.numCols + 1
        self.gamma = 0.5        # time-weighting factor for Crank-Nicholson central-differencing scheme
        print('Read model parameters.')

    def Matrix(self, aquifer, connection, node, Q, area, S, h, dt):

        # assemble the flow model matrix
        diagonal = zeros(self.numNodes, float)
        b = zeros(self.numNodes, float)
        conduct = connection.KMean * area / (connection.d0 + connection.d1)
        data = self.gamma * conduct                      # upper off-diagonal
        rowIndex = connection.node0
        colIndex = connection.node1
        data = concatenate([data, self.gamma * conduct])      # lower off-diagonal; symmetric matrix
        rowIndex = concatenate([rowIndex, connection.node1])
        colIndex = concatenate([colIndex, connection.node0])

        # diagonal terms and explicit matrix        
        for i in range(self.numNodes):
            diagonal[i] = -S[i]/dt
            b[i] = -Q[i]                        # source term
        
        for i in range(len(connection.node0)):
            diagonal[connection.node0[i]] -= self.gamma*conduct[i]
            diagonal[connection.node1[i]] -= self.gamma*conduct[i]            
            b[connection.node0[i]] += conduct[i] * (h[connection.node0[i]] - h[connection.node1[i]])
            b[connection.node1[i]] += conduct[i] * (h[connection.node1[i]] - h[connection.node0[i]])

        # append diagonal, rowIndex and colIndex
        data = concatenate([data, diagonal])
        indices = arange(0, self.numNodes, 1)
        rowIndex = concatenate([rowIndex, indices])
        colIndex = concatenate([colIndex, indices])
        return data, rowIndex, colIndex, b


class Aquifer:      # distribution of subsurface properties, numbered from bottom to top
    
    def __init__(self, model):
        layers = read_csv('layers.csv')
        self.Kh = array(layers['Kh']).astype(float)               # arrays of hydraulic conductivities (1 per layer)
        self.Kv = array(layers['Kv']).astype(float) 
        self.Ss = array(layers['Ss']).astype(float)               # specific storage array
        self.Sy = array(layers['Sy']).astype(float)               # specific yield array
        self.bottom = array(layers['bottom']).astype(float)       # layer structure
        self.top = array(layers['top']).astype(float)       
        self.b = self.top - self.bottom             # layer thickness (large value for top layer if unconfined)
        convert = (model.h0<self.top)               # unconfined initial condition (T/F)
        self.T = self.Kh * ((model.h0-self.bottom)*(convert==True) + self.b*(convert==False))
        self.S = self.Sy*(convert==True) + (self.Ss/self.b)*(convert==False)
        self.layerList = list(arange(0, len(self.Kh)))      # for use in list comprehension
        print('Read aquifer properties.')

    def CooperJacob(self, model):
        # rough estimate of pumping rates to use for small drawdown near well
        s = 0.75 * self.b            # reasonable drawdown at well for scale-up
        u = (model.wellRadius**2 * self.S) / (4. * self.T * model.tEnd/2.)
        W = -0.5772 - log(u)        
        Q = 4. * s * pi * self.T / W
        return -Q
         
        
### supporting & utility functions ###


def DistribTrans(zTop, zBase, aquifer, model):    
    # partitioning of pumping across layers
    zTopFix = (zTop>model.h0)*model.h0 + (zTop<=model.h0)*zTop
    T = ((zTopFix>aquifer.top)*aquifer.top + (zTopFix<=aquifer.top)*zTopFix) \
        - ((zBase>aquifer.bottom)*zBase + (zBase<=aquifer.bottom)*aquifer.bottom)
    T = T * (T>0) * aquifer.Kh
    return T


def HarMeanWt(K0, K1, d0, d1):
    # weighted harmonic means for connections
    return (d0 + d1)/(d0/K0 + d1/K1)


def WellImpact(r, t, Q, Qref, f, model, aquifer, T, TTotal):
    # pumping impact (drawdown per layer) of a single well at distance r and time t
    Qeff = Q * T/TTotal         # distributed pumping
    rQ = Qeff/Qref                                  # multiplier for head correction across each layer
    sLayer = zeros(model.numLayers, float)
    for iMonitor in range(model.numLayers):
        s = [f[j][iMonitor](r, t) * rQ[j] for j in aquifer.layerList]
        sLayer[iMonitor] = sum(s)
    return sLayer


def StackWells(x, y, t, well, Qref, f, model, aquifer, T, TTotal):
    # StackWells effects of all pumping wells, per layer, at (x, y, t); return as drawdown
    r = sqrt((x-well['x'])**2 + (y-well['y'])**2)
    tP = t-well['tStart']
    tR = t-well['tEnd']
    s = zeros(model.numLayers, float) 
    for i in range(len(well)):
        if tP[i] > 0.:
           s += WellImpact(r[i], tP[i], well['Q'].iloc[i], Qref, f, model, aquifer, T[i], TTotal[i])
        if tR[i] > 0.:
           s += WellImpact(r[i], tR[i], -well['Q'].iloc[i], Qref, f, model, aquifer, T[i], TTotal[i]) 
    return s


def Numerical(model, connection, node, aquifer, Qref, h):
    # run numerical cylindrical flow model to extract interpolators
    f = []          # interpolator array; f[pumping layer][monitor layer]
    for layer in range(model.numLayers):
        print('Running numerical model for layer', layer)
        fLayer = []
        Q = zeros(model.numNodes, float)
        Q[layer*model.numCols] = Qref[layer]
        t = 0.
        dt = model.dtMin
        while t < model.tEnd:
            # solver routine
            converged = False
            while converged == False:
                area = connection.FindArea(h)
                S = node.FindStorage(h)
                data, rowIndex, colIndex, b = model.Matrix(aquifer, connection, node, Q, area, S, h, dt)
                A = csr_matrix( (array(data),(array(rowIndex), array(colIndex))), shape=(model.numNodes, model.numNodes) )
                dh = spsolve(A,b)       # solve equations 
                converged = 1 - sign(sum((abs(dh) > model.dhMax) * 1))
                if not converged: dt *= model.dtMinus
                assert(dt >= model.dtMin)    
            # append results to summary dataframe
            h += dh            
            stepResults = DataFrame(data={'layer':node.layer, 'r':node.r,
                't':full(model.numNodes, t+dt), 'h':h, 'delta_h':h-model.h0})
            if t == 0.: currentLayer = stepResults.copy()
            else: currentLayer = concat([currentLayer, stepResults.copy()])
            # update time step
            t += dt
            dt *= model.dtPlus
            dt = min(dt, model.dtMax, model.tEnd-t)    
        # write results to file for inspection
        currentLayer.to_csv('layer_' + str(layer) + '.csv', index=False)        
        # create linear interpolators for pumping from this layer
        for i in range(model.numLayers):
            x = array(currentLayer['r'][currentLayer['layer']==i])
            y = array(currentLayer['t'][currentLayer['layer']==i])
            z = array(currentLayer['delta_h'][currentLayer['layer']==i])
            fLayer.append(interpolate.LinearNDInterpolator((x, y), z))
        f.append(fLayer)
        # reset initial heads
        h = array(full(model.numNodes, model.h0))    
    return f


### main script ###


def SAEM(gridMode, monMode):

    # model attributes
    model = Model()

    # aquifer properties and reference pumping rate
    aquifer = Aquifer(model)
    Qref = aquifer.CooperJacob(model)
    print('Reference pumping rates = ', Qref)
    
    # set up numerical radial flow model
    connection = Connection(model, aquifer)
    node = Node(model, aquifer, connection)

    # write grid data to external files for reference
    h = array(full(model.numNodes, model.h0))    # initial heads
    node.WriteNodes(h)
    connection.WriteConnects(h)    
    
    # run numerical model and assign interpolators
    f = Numerical(model, connection, node, aquifer, Qref, h)
    
    # read grid and output time slice attributes
    grid = Grid()

    # read in monitoring point locations
    monitor = read_csv('monitors.csv')
    print('Read monitoring points.')

    # read well locations, screen intervals, and pumping histories; compuet transmissivities
    well = read_csv('wells.csv')
    T = zeros((len(well), model.numLayers), float)
    TTotal = zeros(len(well), float)
    for i in range(len(well)):
        T[i, :] = DistribTrans(well['zTop'].iloc[i], well['zBase'].iloc[i], aquifer, model)
        TTotal[i] = sum(T[i, :])
    wellList = list(well['name'])
    print('Read well attributes.')

    # check that implied drawdowns will not dewater any layers
    print('Checking implied drawdowns ...')        
    for i, t in enumerate(grid.tMonOut):
        s = zeros((len(well), model.numLayers), float)
        for j in range(len(well)):                                     # find implied drawdowns at pumping wells
            s[j] = StackWells(well['x'].iloc[j], well['y'].iloc[j], t, well, Qref, f, model, aquifer, T, TTotal)
        dewater = model.h0 + s.min(axis=1)      # implied groundwater elevation, by layer 
        failIndxs = [j for j in range(len(well)) if dewater[j] < aquifer.bottom[-1]]
        failWells = [well['name'].iloc[j] for j in failIndxs]
        if len(failWells)>0:
            print('Unsustainable pumping rate or incorrect screen placement. See the following:')
            print('t =', t, 'wells =', failWells)
            sys.exit()
    print('... pumping rates and well screen placements are sustainable.')

    # process grid, by time step
    if gridMode:
        for t in grid.tOut:
            print('Processing grid; t =', t)
            # process each grid point
            gwe = full((len(grid.x), model.numLayers), model.h0)
            for i in range(len(grid.x)):
                gwe[i] += StackWells(grid.x[i], grid.y[i], t, well, Qref, f, model, aquifer, T, TTotal)
            d = {'x':grid.x, 'y':grid.y, 't':full(len(grid.x), t)}
            for j in range(model.numLayers):
                d['layer' + str(j)] = gwe[:, j]
            sliceResults = DataFrame(data=d)
            if t == grid.tOut[0]: gridSummary = sliceResults.copy()
            else: gridSummary = concat([gridSummary, sliceResults.copy()])
        gridSummary.to_csv('GridSummary.csv', index=False)      # write to summary file

    # now sweep through monitor points once; with new hWell values; non-viable wells will drop out
    if monMode:

        # step through time series for each monitoring point        
        for i in range(len(monitor)):
            print('Processing monitor points; location =', monitor['name'].iloc[i])
            head = []
            for j, t in enumerate(grid.tMonOut):
                # process each monitor point
                gwe = model.h0 + StackWells(monitor['x'].iloc[i], monitor['y'].iloc[i], t, well, Qref, f, model, aquifer, T, TTotal)
                head.append(gwe)
                
            d = {'well':[monitor['name'].iloc[i]]*len(grid.tMonOut), 'x':full(len(grid.tMonOut), monitor['x'].iloc[i]), 
                'y':full(len(grid.tMonOut), monitor['y'].iloc[i]), 't':grid.tMonOut}
            head = transpose(head)
            for j in range(model.numLayers):
                d['layer' + str(j)] = head[j]
            monResults = DataFrame(data=d)
            if i == 0: monitorSummary = monResults.copy()
            else: monitorSummary = concat([monitorSummary, monResults.copy()])
            
        # process results
        monList = list(monitor['name'])
        colorList = ['red', 'magenta', 'green' ,'blue', 'cyan']
        for i, monPt in enumerate(monList):
            plt.figure(i)
            for j in range(model.numLayers):
                layerName = 'layer' + str(j)
                x = monitorSummary[monitorSummary['well']==monPt]['t']
                y = monitorSummary[monitorSummary['well']==monPt][layerName]
                plt.plot(x, y, color=colorList[j], label=layerName)
            plt.xlabel('Time')
            plt.ylabel('Head')
            plt.title(monPt)
            plt.legend(loc=3)
            plt.show()        
        monitorSummary.to_csv('MonitorSummary.csv', index=False)    # write to summary file
    
    print('Done.')
    

### run script ###

gridMode = False
monMode = True
SAEM(gridMode, monMode)
