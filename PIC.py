import numpy as np
from scipy import constants
from scipy import interpolate
import csvSaver

class ParticleGenerater:
    k = np.float64(constants.Boltzmann)
    me = np.float64(constants.electron_mass)
    mi = np.float64(constants.proton_mass)

    @staticmethod
    def getElectronLocation (L, N) -> np.ndarray:
        return L*np.random.rand(N)
    
    @staticmethod
    def getIonLocation (L, N) -> np.ndarray:
        return L*np.random.rand(N)

    @staticmethod
    def getElectronVelocity (T, N) -> np.ndarray:
        ev_stdev = np.sqrt((ParticleGenerater.k * T) / (ParticleGenerater.me * N))     # electron superParticle velocity standard deviation
        return np.random.normal(0, ev_stdev, N)      # return np.ndarray
    
    @staticmethod
    def getIonVelocity (T, N) -> np.ndarray:
        iv_stdev = np.sqrt((ParticleGenerater.k * T) / (ParticleGenerater.mi * N))     # ion(proton) superParticle velocity standard deviation
        return np.random.normal(0, iv_stdev, N)      # return np.ndarray

# 1-dimensional
class CumulativeInterpolater:
    @staticmethod
    def linearInterpolate(location_array:np.ndarray, grid_length, grid_number) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_length = np.float64(grid_length)
        count = np.zeros(grid_number-1)
        sRDC = np.zeros(grid_number-1)      # Sum of Relative Distance in Cell
        pRDC = np.copy(location_array)

        particleInCell = (location_array/grid_length).astype(int)
        unique, counts = np.unique(particleInCell, return_counts=True)

        cellCountInform = dict(zip(unique,counts))

        index = 0
        for i in cellCountInform.keys():
            count[i] += cellCountInform[i]
            for j in range(cellCountInform[i]):
                pRDC[index] -= i*grid_length
                sRDC[i] += location_array[index]
                index += 1
            sRDC[i] -= i * cellCountInform[i] * grid_length
        
        # interpolate particles to grids
        result = np.zeros(grid_number)
        result[1:-1] = sRDC[0:-1] - sRDC[1:] + grid_length * count[1:]
        result[0] = count[0] * grid_length - sRDC[0]
        result[-1] = sRDC[-1]

        # devide by grid-length
        result = result/grid_length

        return (pRDC, count, result)
    
    @staticmethod
    def elecToParticle(gridLength: np.float64, gridElectricField: np.ndarray, pRDC: np.ndarray, count: np.ndarray) -> np.ndarray:
        electricFieldL = np.zeros(pRDC.size)
        electricFieldR = np.zeros(pRDC.size)

        index = 0
        for i, num in enumerate(count):
            num = int(num)
            if (num == 0):
                pass
            electricFieldL[index:index+num] = gridElectricField[i]
            electricFieldR[index:index+num] = gridElectricField[i+1]
            index += num

        particleElectricField = ((gridLength - pRDC)*electricFieldL + pRDC*electricFieldR) / gridLength
        """
        print(electricFieldL)
        print(electricFieldR)
        print(gridLength)
        print(pRDC)
        print(particleElectricField)
        """
        return particleElectricField
    
    @staticmethod
    def potentialToParticle(gridLength: np.float64, gridPotentialField: np.ndarray, pRDC: np.ndarray, count: np.ndarray) -> np.ndarray:
        potentialFieldL = np.zeros(pRDC.size)
        potentialFieldR = np.zeros(pRDC.size)

        index = 0
        for i, num in enumerate(count):
            num = int(num)
            if (num == 0):
                pass
            potentialFieldL[index:index+num] = gridPotentialField[i]
            potentialFieldR[index:index+num] = gridPotentialField[i+1]
            index += num

        particlePotentialField = ((gridLength - pRDC)*potentialFieldL + pRDC*potentialFieldR) / gridLength

        return particlePotentialField
            
    
class PoissonSolver:
    def __init__(self, gridNum, gridLength):
        self.gridNum = gridNum
        self.gridLength = gridLength

        temp = np.zeros((gridNum-2, gridNum-2)) # 양 끝점 제외
        for i in range(gridNum-3):
            temp[i][i] = 2
            temp[i][i+1] = -1
            temp[i+1][i] = -1
        temp[gridNum-3][gridNum-3] = 2
        
        # 역행렬 계산
        self.invC: np.ndarray = np.linalg.inv(temp) * (gridLength**2) / constants.epsilon_0

    def fdm1D(self, chargeDensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        # Boundary 는 0으로 가정하자.
        # 코드 리팩토링할때 고려해보자. 일단 구현만
        potential = np.zeros(self.gridNum)
        electricField = np.zeros(self.gridNum)
        potential[1:-1] = ((self.invC).dot(chargeDensity[1:-1].reshape((-1, 1)))).flatten()

        electricField[1:-1] = (potential[0:-2] - potential[2:]) / (2*self.gridLength)
        electricField[0] = ((potential[0] - potential[1]) / self.gridLength) - chargeDensity[0]*self.gridLength / (2 * constants.epsilon_0)
        electricField[-1] = ((potential[-2] - potential[-1]) / self.gridLength) + chargeDensity[-1]*self.gridLength / (2 * constants.epsilon_0)

        return (potential, electricField)

class PIC:
    """
    L -> Length of world
    A -> Area of world
    W -> Superparticle Weight
    Ne -> Num of Electrons (Superparticles)
    Ni -> Num of Ions (Superparticles)
    """
    def __init__(self, T, L, A, GN, W, Ne, Ni) -> None:
        # Spatial
        self.L = np.float64(L)      # World Length
        self.A = np.float64(A)      # World Area
        self.GN = int(GN)    # Number of Grids
        self.grid_length = np.float64(L/(GN-1))     # Grid Length
        self.grid_coordinate = np.mgrid[0:L:complex(0,GN)]     # Coordinate of grids

        # Particles
        self.T = np.float64(T)      # World Temperature
        self.W = int(W)             # Superparticle Weight
        self.Ne = int(Ne)           # Num of Electrons (SuperParticles)
        self.Ni = int(Ni)           # Num of Ions (SuperParticles)
        self.qe = -constants.e
        self.me = constants.m_e
        self.ie = constants.e
        self.mi = constants.m_p

        # Information of Particles
        self.p_e = None
        self.r_e = None
        self.v_e = None
        self.p_i = None
        self.r_i = None
        self.v_i = None

        # to Save
        self.inform_gridChargeDenstiy = []
        self.inform_electronLocation = []
        self.inform_electronVelocity = []
        self.inform_systemTotalEnergy = []

        # Util Class
        self.PoisonSolver = PoissonSolver(self.GN, self.grid_length)
    
    def rvSort(self, r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(r)
        r = r[idx]
        v = v[idx]
        return (r,v)
    
    def rvSort(self, r: np.ndarray, v: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.argsort(r)
        r = r[idx]
        v = v[idx]
        p = p[idx]
        return (r,v,p)

    def generateParticles(self) -> None:
        self.p_e = np.arange(0,self.Ne)
        self.r_e = ParticleGenerater.getElectronLocation(L = self.L, N = self.Ne)
        self.v_e = ParticleGenerater.getElectronVelocity(T = self.T, N = self.Ne)
        self.r_e, self.v_e, self.p_e = self.rvSort(self.r_e, self.v_e, self.p_e)

        self.p_i = np.arange(0,self.Ni)
        self.r_i = ParticleGenerater.getIonLocation(L = self.L, N = self.Ni)
        self.v_i = ParticleGenerater.getIonVelocity(T = self.T, N = self.Ni)
        self.r_i, self.v_i, self.p_i = self.rvSort(self.r_i, self.v_i, self.p_i)
    
    def simulationStart(self, totalTime, timeStep) -> None:
        iterations = int(totalTime/timeStep)
        print(self.r_e)
        print(self.v_e)
        # 시뮬레이션 시작
        # 1. Interpolate particles(charges) to grids
        # 2. Calculate E by FDM
        # 3. Interpolate grids(forces) to particles
        # (3-5.) Data Save
        # 4. Move Particle
        # 5. Sort Particle
        for i in range(iterations):
            # 1. Interpolate particles(charge density) to grids
            e_pRDC, e_count, e_interpolated = CumulativeInterpolater.linearInterpolate(self.r_e, self.grid_length, self.GN)
            i_pRDC, i_count, i_interpolated = CumulativeInterpolater.linearInterpolate(self.r_i, self.grid_length, self.GN)
            gridChargeDensity = constants.e * (i_interpolated - e_interpolated) * (self.W / (self.A * self.grid_length))

            # 2. Calculate E by FDM
            (gridPotential, gridElectricField) = self.PoisonSolver.fdm1D(gridChargeDensity)

            # 3. Interpolate grids(E) to particles
            e_particleElectricField = CumulativeInterpolater.elecToParticle(self.grid_length, gridElectricField, e_pRDC, e_count)
            i_particleElectricField = CumulativeInterpolater.elecToParticle(self.grid_length, gridElectricField, i_pRDC, i_count)

            # 3.5 Interpolate grids(V(Potential)) to particles
            # 3.5. iteration save
            if i%5 == 0:
                e_particlePotential = CumulativeInterpolater.potentialToParticle(self.grid_length, gridPotential, e_pRDC, e_count)
                i_particlePotential = CumulativeInterpolater.potentialToParticle(self.grid_length, gridPotential, i_pRDC, i_count)
                energy = np.sum(0.5*self.me*(self.v_e**2) + 0.5*self.mi*(self.v_i**2) + self.qe*e_particlePotential + self.ie*i_particlePotential)
                self.inform_systemTotalEnergy.append([energy])
                """
                r_temp = [0,0,0,0,0]
                v_temp = [0,0,0,0,0]
                temp_count = 0
                for real, indx  in enumerate(self.p_e):
                    if indx == 0 or indx == 1 or indx == 2 or indx == 3 or indx == 4:
                        r_temp[indx] = self.r_e[real]
                        v_temp[indx] = self.v_e[real]
                        temp_count += 1
                        if temp_count == 5:
                            break
                self.inform_electronLocation.append(r_temp)
                self.inform_electronVelocity.append(v_temp)
                self.inform_gridChargeDenstiy.append(gridChargeDensity)
                """
            
            # 4. Move Particles
            self.v_e += (self.qe / self.me) * e_particleElectricField * timeStep
            self.v_i += (self.ie / self.mi) * i_particleElectricField * timeStep

            self.r_e += self.v_e * timeStep
            self.r_i += self.v_i * timeStep

            # 5. Sort
            self.r_e, self.v_e, self.p_e = self.rvSort(self.r_e, self.v_e, self.p_e)
            self.r_i, self.v_i, self.p_i = self.rvSort(self.r_i, self.v_i, self.p_i)

            # 6. Boundary check
            for index,r in enumerate(self.r_e):
                if (r < 0):
                    self.r_e[index] += self.L * (1 - int(r/self.L))
                else :
                    break
            for index,r in reversed(list(enumerate(self.r_e))):
                if (r >= self.L):
                    self.r_e[index] -= self.L * (int(r/self.L))
                else :
                    break
            
            for index,r in enumerate(self.r_i):
                if (r < 0):
                    self.r_i[index] += self.L * (1 - int(r/self.L))
                else :
                    break
            for index,r in reversed(list(enumerate(self.r_i))):
                if (r >= self.L):
                    self.r_i[index] -= self.L * (int(r/self.L))
                else :
                    break
            
            # 7. ReSort
            self.r_e, self.v_e, self.p_e = self.rvSort(self.r_e, self.v_e, self.p_e)
            self.r_i, self.v_i, self.p_i = self.rvSort(self.r_i, self.v_i, self.p_i)
        
        #csvSaver.listToCsv(self.inform_electronLocation, "test_loc.csv")
        #csvSaver.listToCsv(self.inform_electronVelocity, "test_vel.csv")
        #csvSaver.listToCsv(self.inform_gridChargeDenstiy, "test_GCD.csv")
        csvSaver.listToCsv(self.inform_systemTotalEnergy, "test_Energy.csv")
            
a = PIC(T=30, L=10, A=0.08, GN=501, W=1000, Ne=2000, Ni=2000)
a.generateParticles()
a.simulationStart(1e-5,1e-9)


#print(CumulativeInterpolater.linearInterpolate(np.array([1]), 1, 11))