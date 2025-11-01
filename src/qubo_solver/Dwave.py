from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimod import Binary

def dwave_solve(Q,token):
  """
  token: API token from D-Wave
  """
  qpu = DWaveSampler(token=token)
  sampler = EmbeddingComposite(qpu)
  
  Q = Q.detach().numpy()
  qubo = 0
  for i in range(len(Q)):
      qubo += Q[i,i]*Binary(i)
      for j in range(i+1,len(Q)):
          qubo += 2*Q[i,j]*Binary(i)*Binary(j)

  sampleset = sampler.sample(qubo,num_reads=100)
  best_sample = sampleset.first.sample
  solution = [best_sample[i] for i in range(len(Q))]
  energy = sampleset.first.energy

  return energy,solution
