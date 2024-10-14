import numpy as np
import scipy.special
import scipy.misc
import csv
import math
from tqdm import tqdm

from collections import defaultdict

# Responsible for running a single Concept / Restaurant
def parallel(parameters):
    iterations = 500 # originally 500
    burnIn = 100
    maxTrials = 36

    uniformPrior = parameters[0]
    alpha = parameters[1]
    chain = parameters[2]
    concept = parameters[3]

    result = ([], [])

    # print(concept[0], "Trials", maxTrials, "Participants", len(concept[1]), "Alpha", alpha, "Chain", chain, "Uniform Prior", uniformPrior)
    sampler = Gibbs(concept[1], iterations, burnIn, uniformPrior, maxTrials, alpha, chain)
    sampler.run(concept[0], result)

    return result

def lgamma(x):
    return scipy.special.gammaln(x)

def logfactorial(x):
    return lgamma(x + 1)

def betaBernoulliLikelihood(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def logsumexp(x):
    return scipy.special.logsumexp(x)

# A person is a vector of responses, 1 for each question
class Person:
    def __init__(self, id, r):
        self.id = id
        self.responses = r

    def __str__(self):
        return "[P%s]" % ''.join(map(str, self.responses))

# A table consists of 1 or more people
class Table:
    members = []

    def __init__(self, members, alpha):
        self.members = members
        self.alpha = alpha

    def add(self, p):
        self.members.append(p)

    # Remove person from table, if they're there
    def remove(self, p):
        if p in self.members:
            self.members.remove(p)

    # Returns the number of people at the table
    def __len__(self):
        return len(self.members)

    # The likelihood for this table alone
    def likelihood(self):
        # Total number of people in the table
        n = len(self.members)

        out = 0.0
        # Loop through all questions
        for q in range(len(self.members[0].responses)):

            # Make sure everyone is the same length
            for p in self.members: assert (len(p.responses) == len(self.members[0].responses))

            # Number of yes answers
            yes = sum([p.responses[q] == 1 for p in self.members])

            # Number of no answers
            no = sum([p.responses[q] == 0 for p in self.members])

            assert (yes + no == n)

            # Compute the likelihood
            out += (betaBernoulliLikelihood(yes + self.alpha, no + self.alpha) - betaBernoulliLikelihood(self.alpha, self.alpha))
        return out

# A restaurant has zero or more tables
class Restaurant:
    def __init__(self, population, temperature, alpha):  # initialize with everyone at the same table
        self.tables = []
        self.temperature = temperature
        self.alpha = alpha
        self.population = population

        for p in population:
            self.tables.append(Table([p], self.alpha))

    # Total number of people in the restaurant
    def Npeople(self):
        return sum([len(t) for t in self.tables])

    # Total number of tables in the restaurant
    def Ntables(self):
        return len(self.tables)

    # Total number of tables with N people
    def tablesWithNPeople(self, number):
        tables = 0

        for t in self.tables:
            if t.__len__() == number:
                tables += 1

        return tables

    def sizeOfSmallestTable(self):
        min = math.inf

        for t in self.tables:
            if t.__len__() < min:
                min = t.__len__()

        return min

    def sizeOfLargestTable(self):
        max = 0

        for t in self.tables:
            if t.__len__() > max:
                max = t.__len__()

        return max

    def computeSum(self):
        totalPeople = self.Npeople()
        x = 0
        people = 1

        while totalPeople != 0:
            tablesWith = self.tablesWithNPeople(people)
            totalPeople -= tablesWith * people

            x += np.sum(people * (people - 1) * tablesWith)
            people += 1

        return x

    # Remove this person from any/all tables and delete the empty tables
    def remove(self, p):
        for t in self.tables:
            t.remove(p)

        self.tables = list(filter(lambda t: len(t) > 0, self.tables))  # toss anything empty

    # Which table is p sitting at?
    def where(self, p):
        for i in range(len(self.tables)):
            if p in self.tables[i].members:
                return i
        return None

    # Add person to either a new table, or an existing table
    def seat_at(self, p, i):
        if (i >= len(self.tables)):
            assert (i == len(self.tables))
            self.tables.append(Table([p], self.alpha))
        else:
            self.tables[i].add(p)

    # Compute likelihood of the entire restaurant
    def likelihood(self):
        return sum([t.likelihood() for t in self.tables])

    # Compute prior of the entire restaurant
    def CRP_prior(self, uniformPrior):
        out = 0.0
        constant = 0.0

        if not(uniformPrior):
            constant = (lgamma(1) + self.Ntables() * np.log(1)) - lgamma(self.Npeople() + 1)

            for t in self.tables:
                out += lgamma(len(t))

        return constant + out

    # Compute posterior of the entire restaurant
    def posterior(self, uniformPrior, temperature = None):
        if (temperature is None):
            temperature = self.temperature

        return (self.CRP_prior(uniformPrior) + self.likelihood()) / temperature

    # Print out details of restaurant
    def show(self):
        for i, t in enumerate(self.tables):
            print("Table ", i, "\t", "Size ", len(t), "\t", ' '.join([str(p) for p in t.members]))
        print("------------------")

# Contains people and their responses from a dataset
class Data:
    def __init__(self, numberOfTrials, maxNumberOfParticipants, filename):
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "UH OH")))

        self.people = defaultdict(lambda: [])
        self.numberOfTrials = numberOfTrials

        # Reads CSV file that contains responses from behavioral experiment
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')

            for i, row in enumerate(reader):
                if i != 0:
                    # Concept -> Participant -> Trial = Response
                    self.data[row[0]][row[1]][row[2]] = int(row[3])

        # Loop through all concepts
        for conceptName, concept in self.data.items():
            numberOfParticipants = 0

            # Loop through all participants
            for participantID, participant in self.data[conceptName].items():
                id = participantID
                responses = []

                # Loop through all trials
                for trial, response in self.data[conceptName][participantID].items():
                    responses.append(response)

                    if len(responses) == self.numberOfTrials:
                        break

                # Only include a participant if they responded to all trials
                if len(responses) == self.numberOfTrials:
                    numberOfParticipants += 1
                    self.people[conceptName].append(Person(id, responses))

                if numberOfParticipants == maxNumberOfParticipants:
                    break

# Samples from Chinese Restaurant Process
class Gibbs:
    def __init__(self, people, iterations, burnIn, uniformPrior, maxTrials, alpha, chain):
        self.restaurants = [Restaurant(people, 1.0, alpha)]
        self.posterior_count = defaultdict(int)
        self.uniqueConcepts = defaultdict(int)
        self.iterations = iterations
        self.burnIn = burnIn

        self.numPeople = len(people)
        self.numberOfTrials = maxTrials
        self.uniformPrior = uniformPrior
        self.alpha = alpha
        self.chain = chain

    def run(self, conceptName, result):
        MAP = float("-inf")
        MAPPopulation = []

        # For each iteration, each restaurant, and each person...
        for iteration in tqdm(range(self.iterations)):
            for r in self.restaurants:
                for person in r.population:
                    scores = []
                    #person = np.random.choice(r.population, size = 1)[0]

                    r.remove(person)

                    # Place person in each table and calculate posterior
                    for table in r.tables:
                        table.add(person)
                        scores.append(r.posterior(self.uniformPrior))
                        table.remove(person)

                    # Calculate posterior for person in a new table
                    r.seat_at(person, r.Ntables())
                    scores.append(r.posterior(self.uniformPrior))
                    r.remove(person)

                    # Determining winning table
                    winningTableIndex = np.random.choice(r.Ntables() + 1, p = np.exp(scores - logsumexp(scores)))
                    r.seat_at(person, winningTableIndex)

            # If the burn-in period is over, calculate estimator and output results
            if iteration > self.burnIn:
                self.posterior_count[self.restaurants[0].Ntables()] += 1

                tablesWithOne = self.restaurants[0].tablesWithNPeople(1)
                tablesWithTwo = self.restaurants[0].tablesWithNPeople(2)
                tablesWithThree = self.restaurants[0].tablesWithNPeople(3)
                tablesWithFour = self.restaurants[0].tablesWithNPeople(4)

                if self.restaurants[0].tablesWithNPeople(2) > 0:
                    x = (tablesWithOne ** 2) / (2 * tablesWithTwo)
                else:
                    x = tablesWithOne * (tablesWithOne - 1) / 2

                # ecological estimator calculations (304-309)
                chao1 = self.restaurants[0].Ntables() + ((self.numPeople - 1) / self.numPeople) * x
                N = chao1

                if (tablesWithFour > 0):
                    N = chao1 + ((self.numPeople - 3) / self.numPeople) * (tablesWithThree / (4 * tablesWithFour)) * \
                        max(tablesWithOne - ((self.numPeople - 3) / (self.numPeople - 1)) * ((tablesWithTwo * tablesWithThree) / (2 * tablesWithFour)), 0)

                self.uniqueConcepts[N] += 1

                uniformPriorOutput = "Simplicity"

                if self.uniformPrior:
                    uniformPriorOutput = "Uniform"

                posterior = self.restaurants[0].posterior(self.uniformPrior)

                # Calculate MAP configuration but only if this restaurant is our main one
                if (posterior > MAP) and (self.alpha == .16) and (self.uniformPrior == False) and (self.chain == 1) and (self.numPeople >= 80):
                    MAP = posterior
                    MAPPopulation = []
                    for person in self.restaurants[0].population:
                        MAPPopulation.append([person.id, self.restaurants[0].where(person), conceptName])

                probabilityOfSameTable = 0

                for table in self.restaurants[0].tables:
                    probabilityOfSameTable += (table.__len__() / self.numPeople) ** 2

                result[0].append([conceptName, iteration, N, self.numPeople, self.numberOfTrials, uniformPriorOutput,
                                  self.restaurants[0].Ntables(), self.alpha, posterior, self.chain, probabilityOfSameTable])

                # Output MAP configuration results only if this restaurant is our main one
                if (iteration == self.iterations - 1) and (self.alpha == .16) and (self.uniformPrior == False) and (self.chain == 1) and (self.numPeople >= 80):
                    for person in MAPPopulation:
                        result[1].append([person[0], person[1], person[2]])
