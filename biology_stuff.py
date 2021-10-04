#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Dict, List
import math

def Counting_DNA_Nucleotides(DNA: str = "ACGT"):
  # ----------------------------------------------------------------
  A: int = 0
  C: int = 0
  G: int = 0
  T: int = 0
  for letter in DNA:
    if letter == 'A': A += 1
    if letter == 'C': C += 1
    if letter == 'G': G += 1
    if letter == 'T': T += 1
  print (f"{A} {C} {G} {T}")
  # ----------------------------------------------------------------

  # count is a low level C method so it is faster, but we'have to run through the sequence 4 times
  print(DNA.count("A"), DNA.count("G"), DNA.count("C"), DNA.count("T"))
  
  return None

def Transcribing_DNA_into_RNA(codingStrand: str = "ACGT"):

  RNA: str = codingStrand.replace("T", "U")
  print(RNA)

  return None

def Complementing_Strand_of_DNA(DNA: str = "AAAACCCGGT"):
  # reverse complement, DNA_c: AAAACCCGGT -> ACCGGGTTTT

  # # reverse
  # DNA_c = list(DNA[::-1])
  # # replace
  # for i in range(len(DNA_c)):
  #   if DNA_c[i] == "A": DNA_c[i] = "T"
  #   elif DNA_c[i] == "T": DNA_c[i] = "A"
  #   elif DNA_c[i] == "C": DNA_c[i] = "G"
  #   elif DNA_c[i] == "G": DNA_c[i] = "C"
  # # join list back into str
  # DNA_c = "".join(DNA_c)
  
  complement = { "A" : "T", "T" : "A", "C" : "G", "G" : "C"}
  print("".join([complement[i] for i in DNA[::-1]]))

  return None

def recursive_fib(n: int):
  if n == 0: return 0
  elif n == 1: return 1
  else: return fib(n - 1) + fib(n - 2)

def iterative_fib(n: int):
  # -------------- iterative solution
  n0, n1, nth = 1, 0, 0 # starting terms
  for _ in range(n, 0, -1):
    # count fibonacci term
    nth =  n1 + n0
    # update previous terms
    n0 = n1
    n1 = nth
  return nth

def Fibonacci_Rabbits(n: int = 30, k: int = 5, i: int = 1):
  # with i = 1, n = 5, k = 3: the answer is 19 pairs of rabits after 5 months
  # n for months / terms
  # k for litter produced per mating 
  # i for num of initial generations. Usually '1'
  
  # ******************************************
  # Fn = Fn−1 + k * Fn−2
  # curr_gen = (previous gen + (previous previous gen * k)) * i
  # ******************************************

  # k = 1: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987
  # k = 2: 0, 1, 1, 3, 5, 11, 21
  # k = 3: 0, 1, 1, 4, 7, 19, 40, 97

  # ---------------------- nth term fibonacci
  # Phi: int = (math.sqrt(5) + 1)/2
  # phi: int = Phi - 1
  # Fn = int((math.pow(Phi, n) - (math.pow(-1, n)/math.pow(Phi, n))) / math.sqrt(5))

  # -------------- recursive func
  # print(recursive_fib(n) * i)

  # -------------- iterative solution
  n0, n1, nth = 0, 1, 0 # starting terms
  while(n > 1):
    # count fibonacci term
    nth =  n1 + (n0 * k)
    # update previous terms
    n0 = n1
    n1 = nth
    # mark the finished term
    n -= 1

  print(i * nth)

  return None

def computing_GC_Content(fasta_file: str = "some_file.fasta"):
  fasta: dict = {}
  target_id: str = ""
  MAX_CG_percentage: float = 0
  CURR_CG_percentage: float = 0

  with open(fasta_file, 'r') as fh:
    for line in fh:
      line = line.strip() # strip it from keading and trialing spaces
      if not line: # bascially checking if string is empty
        continue
      if line.startswith(">"):
        seq_id = line[1:] # grab the entire id 
        if seq_id not in fasta: # add the id to the dictionary
          fasta[seq_id] = []
        continue
      seq = line # this line doesn't start with '>', so it is the seq
      fasta[seq_id].append(seq)

  # fix dictionary if it has mulitple seq per if and find 
  for id, seq in fasta.items():
    if len(seq) > 1 :
      seq = ''.join(seq)
    CURR_CG_percentage = ((seq.count("C") + seq.count("G")) / len(seq)) * 100
    if (CURR_CG_percentage > MAX_CG_percentage):
      target_id = id
      MAX_CG_percentage = CURR_CG_percentage
  
  print(target_id, MAX_CG_percentage)
  return None

def Counting_Point_mutations(s: str = "GAGCCTACTAACGGGAT", t: str = "CATCGTAATGACGGCCT"):
  # s and t must be of equal length, default answer is 7
  # the Hamming distance between s and t, denoted dH(s,t), is the number of corresponding symbols that differ in s and t.

  # makes a list of tuples of all the mismatches (s, t), then prints its length
  print(len([(s[i], t[i]) for i in range(len(s)) if s[i] != t[i]]))

  # alternatives
  # print [ a!=b for (a, b) in zip(s1, s2)].count(True)
  # sum([a != b for a, b in zip(s1, s2)])

  return None

def Mendel_First_Law(k: int = 24, m: int = 20, n: int = 15):
  # Given: Three positive integers k, m, and n, representing a population containing k + m + n organisms: 
  # k individuals are homozygous dominant for a factor, m are heterozygous, and n are homozygous recessive.
  # T is total individuals
  # Return: The probability that two randomly selected mating organisms will produce an individual possessing a dominant allele 
  # (and thus displaying the dominant phenotype). Assume that any two organisms can mate.

  # formal explanation http://rosalind.info/problems/iprb/explanation/
  # 2 2 2 --> 0.78333
  # 24 20 15 --> 0.8232

  T: int = k + m + n
  dominant_allele_produce_probability: float = (k * (k + (2 * m) + (2 * n) - 1) + m * (((3/4) * m) + (n - (3/4)))) / (T * (T - 1))
  print(dominant_allele_produce_probability)

  return None

def Translating_RNA_into_Protein(rna_string: str = "AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA"):
  # default returns: MAMAPRTEINSTRING
  
  # Where "U" = 0, "C" = 1, "A" = 2, "G' = 3

  # total 16 per same first letter in each group
  # total 4 per same second letter in each group
  # total 1 per same third letter in each group

  # 000, 001, 002, 003, 
  # 010, 011, 012, 013,
  # 020, 021, 022, 023,
  # 030, 031, 032, 033.

  # 100, 101, 102, 103, 
  # 110, 111, 112, 113,
  # 120, 121, 122, 123,
  # 130, 131, 132, 133.

  # 200, 201, 202, 203, 
  # 210, 211, 212, 213,
  # 220, 221, 222, 223,
  # 230, 231, 232, 233.

  # 300, 301, 302, 303, 
  # 310, 311, 312, 313,
  # 320, 321, 322, 323,
  # 330, 331, 332, 333.

  codon_table = "FFLLSSSSYY00CC0WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
  nucleotide : dict = {"U" : 0, "C" : 1, "A" : 2, "G" : 3} 
  STOP: str = "0"

  protein_string: str = ""
  for i in range(0, len(rna_string), 3):
    next_protein = codon_table[nucleotide[rna_string[i]] * 16 + nucleotide[rna_string[i+1]] * 4 + nucleotide[rna_string[i+2]] * 1]
    if (next_protein == STOP):
      break
    protein_string += next_protein

  print(protein_string)
  return None

def finding_motif_in_DNA(dna_seq: str = "GATATATGCATATACTT", motif: str = "ATAT"):
  # default returns 2 4 10

  print(" ".join([str(i + 1) for i in range(len(dna_seq)) if dna_seq[i] == motif[0] if (dna_seq[i:len(motif)+i] == motif)]))

  return None

def consensus_and_profile(fasta_file: str = "some_file2.fasta"):
  # [[A count in x strings], [C count in x strings], [G count in x strings], [U count in x strings]]
  # [[5, 1, 0, 0, 5, 5, 0, 0], [0, 0, 1, 4, 2, 0, 6, 1], [1, 1, 6, 3, 0, 1, 0, 0], [1, 5, 0, 0, 0, 1, 1, 6]]
  fasta: dict = {}
  nucleotide_count: list[list[int]] = []
  consensus: str = ""
  total_nucleotides: int = 4
  A: int = 0
  C: int = 1
  G: int = 2
  T: int = 3

  # read in info
  seq_id: int = 0
  with open(fasta_file, 'r') as fh:
    for line in fh:
      line = line.strip() # strip it from keading and trialing spaces
      if not line: # bascially checking if string is empty
        continue
      if line.startswith(">"):
        seq_id += 1
        if seq_id not in fasta: # add the id to the dictionary
          fasta[str(seq_id)] = []
        continue
      seq = line # this line doesn't start with '>', so it is the seq
      fasta[str(seq_id)].append(seq)
  # convert seq to str instead of lists
  for id, seq in fasta.items():
    temp: str = "".join(seq)
    fasta[id] = temp

  # construct matrix
  for a in range(0, total_nucleotides, 1):
    nucleotide_count.append([])
    for b in range(len(fasta["1"])):
      nucleotide_count[a].append(0)
  
  # start the counting
  for j in range(1, len(fasta) + 1, 1):
    for i in range(0, len(fasta[str(j)]), 1):
      if (fasta[str(j)][i] == "A"):
        nucleotide_count[0][i] += 1
      if (fasta[str(j)][i] == "C"):
        nucleotide_count[1][i] += 1
      if (fasta[str(j)][i] == "G"):
        nucleotide_count[2][i] += 1
      if (fasta[str(j)][i] == "T"):
        nucleotide_count[3][i] += 1

  # find the consensus
  for y in range(0, len(fasta["1"]), 1):
    max_value: int = max(nucleotide_count[0][y], nucleotide_count[1][y], nucleotide_count[2][y], nucleotide_count[3][y])
    for x in range(0, total_nucleotides, 1):
      if (nucleotide_count[x][y] == max_value):
        if (x == A): 
          consensus += "A"
          break
        if (x == C): 
          consensus += "C"
          break
        if (x == G): 
          consensus += "G"
          break
        if (x == T): 
          consensus += "T"
          break
      
  # print derived info
  print(consensus)
  print("A:", " ".join([str(integer) for integer in nucleotide_count[A]]))
  print("C:", " ".join([str(integer) for integer in nucleotide_count[C]]))
  print("G:", " ".join([str(integer) for integer in nucleotide_count[G]]))
  print("T:", " ".join([str(integer) for integer in nucleotide_count[T]]))

  return None

def mortal_fibonacchi_rabbits(m: int = 6, n: int = 6, k: int = 1, i: int = 1):
  # m for months / terms
  # n for life expectency, only applicable if mortal
  # k for litter produced per mating 
  # i for num of initial generations. Usually '1'

  #       1,  1, 2, 3, 5, 8, 13, 21, 34, 55
  # n=3   [1, 1, 2, 2, 3, 4, 5, 7, 9, 12]        f(n) = f(n-2) + f(n-3)
  # n=4   [1, 1, 2, 3, 4, 6, 9, 13, 19, 28]      f(n) = f(n-2) + f(n-3) + f(n-4)
  # n=5   [1, 1, 2, 3, 5, 7, 11, 17, 26, 40]     f(n) = f(n-2) + f(n-3) + f(n-4) + f(n-5)
  # n=6   [1, 1, 2, 3, 5, 8, 12, 19, 30, 47]     f(n) = f(n-2) + f(n-3) + f(n-4) + f(n-5) + f(n-6)
  # n=7   [1, 1, 2, 3, 5, 8, 13, 20, 32, 51]     f(n) = f(n-2) + f(n-3) + f(n-4) + f(n-5) + f(n-6) + f(n-7)

  population = [1, 1]
  for i in range(2, m, 1):
    # if we are less than life expectancy, i < n, normal fib
    aux = population[i - 1] + population[i - 2]
    # when we equal life expectancy, subtract 1
    if (i == n): aux = aux - 1
    # when we above life expectancy, we add the terms following the above idea
    if (i > n): aux = aux - population[i - n - 1]
    population.append(aux)

  print(population[-1])
  return None

def printDictionary(dictionary: Dict):
  print()
  # prints a dictionary with *function* values
  for key, value in dictionary.items():
    print(f"{key}. {value.__name__}")
  print()
  return None

def terminate():
  exit()
  return None

options: Dict = {
  '01': Counting_DNA_Nucleotides,
  '02': Transcribing_DNA_into_RNA,
  '03': Complementing_Strand_of_DNA,
  '04': Fibonacci_Rabbits,
  '05': computing_GC_Content,
  '06': Counting_Point_mutations,
  '07': Mendel_First_Law,
  '08': Translating_RNA_into_Protein,
  '09': finding_motif_in_DNA,
  '10': consensus_and_profile,
  '11': mortal_fibonacchi_rabbits
}

def main():
  while(True):
    # print options for the user
    printDictionary(options)
    # pick a feature
    feature = input("What would you like to do (to exit, input a non-option)?_ ")
    # Get the function from options dict and execute it 
    options.get(feature, terminate)()
  return None

if __name__ == "__main__":
  main()