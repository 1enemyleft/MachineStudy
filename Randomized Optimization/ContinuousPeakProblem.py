import sys
import os
import time

sys.path.append("/Users/jiananduan/Documents/ML7641/ABAGAIL/ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
from time import clock
from itertools import product

"""
Commandline parameter(s):
   none
"""

N=100
T=49	
maxIters = 3001
numTrials=5
fill = [2] * N
ranges = array('i', fill)
outfile = './ContinuousPeakProblem/CONTPEAKS_@ALG@_@N@_LOG.txt'
ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# RHC
for t in range(numTrials):
	fname = outfile.replace('@ALG@','RHC').replace('@N@',str(t+1))
	with open(fname,'w') as f:
		f.write('iterations,fitness,time,fevals\n')
	ef = ContinuousPeaksEvaluationFunction(T)
	odd = DiscreteUniformDistribution(ranges)
	nf = DiscreteChangeOneNeighbor(ranges)
	hcp = GenericHillClimbingProblem(ef, odd, nf)
	rhc = RandomizedHillClimbing(hcp)
	fit = FixedIterationTrainer(rhc, 10)
	times =[0]
	fevals = 0 
	for i in range(0,maxIters,10):
		start = clock()
		fit.train()
		elapsed = time.clock()-start
		times.append(times[-1]+elapsed)
		score = ef.value(rhc.getOptimal())
		fevals -= 1
		st = '{},{},{},{}\n'.format(i,score,times[-1],fevals)
		print st	
		with open(fname,'a') as f:
			f.write(st)
 


# SA
for t in range(numTrials):
	for CE in [0.15,0.35,0.55,0.75,0.95]:
		fname = outfile.replace('@ALG@','SA{}'.format(CE)).replace('@N@',str(t+1))
		with open(fname,'w') as f:
			f.write('iterations,fitness,time,fevals\n')
		ef = ContinuousPeaksEvaluationFunction(T)
		odd = DiscreteUniformDistribution(ranges)
		nf = DiscreteChangeOneNeighbor(ranges)
		hcp = GenericHillClimbingProblem(ef, odd, nf)
		sa = SimulatedAnnealing(1E10, CE, hcp)
		fit = FixedIterationTrainer(sa, 10)
		times =[0]
		fevals = 0 
		for i in range(0,maxIters,10):
			start = clock()
			fit.train()
			elapsed = time.clock()-start
			times.append(times[-1]+elapsed)
			score = ef.value(sa.getOptimal())
			fevals -= 1
			st = '{},{},{},{}\n'.format(i,score,times[-1],fevals)
			print st
			with open(fname,'a') as f:
				f.write(st)

#GA
for t in range(numTrials):
	for pop,mate,mutate in product([100],[50,30,10],[50,30,10]):
		fname = outfile.replace('@ALG@','GA{}_{}_{}'.format(pop,mate,mutate)).replace('@N@',str(t+1))
		with open(fname,'w') as f:
			f.write('iterations,fitness,time,fevals\n')
		ef = ContinuousPeaksEvaluationFunction(T)
		odd = DiscreteUniformDistribution(ranges)
		nf = DiscreteChangeOneNeighbor(ranges)
		mf = DiscreteChangeOneMutation(ranges)
		cf = SingleCrossOver()
		gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
		ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
		fit = FixedIterationTrainer(ga, 10)
		times =[0]
		fevals = 0 
		for i in range(0,maxIters,10):
			start = clock()
			fit.train()
			elapsed = time.clock()-start
			times.append(times[-1]+elapsed)
			score = ef.value(ga.getOptimal())
			fevals -= 1
			st = '{},{},{},{}\n'.format(i,score,times[-1],fevals)
			print st
			with open(fname,'a') as f:
				f.write(st)

#MIMIC
for t in range(numTrials):
	for samples,keep,m in product([100],[50],[0.1,0.3,0.5,0.7,0.9]):
		fname = outfile.replace('@ALG@','MIMIC{}_{}_{}'.format(samples,keep,m)).replace('@N@',str(t+1))
		with open(fname,'w') as f:
			f.write('iterations,fitness,time,fevals\n')
		ef = ContinuousPeaksEvaluationFunction(T)
		odd = DiscreteUniformDistribution(ranges)
		nf = DiscreteChangeOneNeighbor(ranges)
		mf = DiscreteChangeOneMutation(ranges)
		cf = SingleCrossOver()
		gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
		df = DiscreteDependencyTree(m, ranges)
		pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
		mimic = MIMIC(samples, keep, pop)
		fit = FixedIterationTrainer(mimic, 10)
		times =[0]
		fevals = 0 
		for i in range(0,maxIters,10):
			start = clock()
			fit.train()
			elapsed = time.clock()-start
			times.append(times[-1]+elapsed)
			score = ef.value(mimic.getOptimal())
			fevals -= 1
			st = '{},{},{},{}\n'.format(i,score,times[-1],fevals)
			print st
			with open(fname,'a') as f:
				f.write(st)


