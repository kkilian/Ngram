import random
class NgramModel(object):
    
    def __init__(self, n):
        self.n = n
        self.context = {}
        self.ngram_counts = {}
    
    def update(self, ngrams):
        for ngram in ngrams:
            ngram = tuple(ngram)
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

            if len(ngram) == 2:
                ngram_context = ngram[0]
            else:
                ngram_context = ngram[:-1]

            self.context[ngram_context] = self.context.get(ngram_context, 0) + 1
        
    def calculate_probs(self):
        self.probs = {}
        for key in self.ngram_counts.keys():
            if self.n == 2:
                self.probs[key] = self.ngram_counts[key] / self.context[(key[:-1][0])] 
            else:
                self.probs[key] = self.ngram_counts[key] / self.context[key[:-1]]

        return self.probs
    def probs(self, context_token, token):
        token_count = self.ngram_counts.get((context_token, token), 0)
        context_count = self.context.get(context_token, 0)
        if context_count == 0:
            result = 0 
        else:
            result = float(token_count / context_count)
        return result
    
    def perplexity(self, game):
        probability = 0
        for i in range(len(game)):
            if i == (self.n - 1):
                ngram = tuple(game[i-(self.n-1):i+1])
                probability = (self.probs.get(ngram, 0))**(-1/i)
                print(f"ngram probability: {self.probs.get(ngram, 0)}, perplexity: {self.probs.get(ngram, 0)}")
            elif i > (self.n - 1):
                ngram = tuple(game[i-(self.n-1):i+1])
                probability *= (self.probs.get(ngram, 0))**(-1/i)
                print(f"ngram probability: {self.probs.get(ngram, 0)}, perplexity: {probability}")
            print(f"perplexity: \033[1m{round(probability, 1)}\033[0m")
   
    def random_token(self, context_token):

            r = random.random()
            map_to_probs = {}
            matching_keys = [key for key in self.ngram_counts.keys() if key[0] == context_token]
            for token in matching_keys:
                map_to_probs[token] = self.probs(context_token, token[1])
                
            summ = 0
            for token in sorted(map_to_probs):
                summ += map_to_probs[token]
                if summ > r:
                    return token
    def generate_text(self, token_count: int):
       
            n = self.n
            context_queue = (n - 1) * ['s']
            result = []
            
            for _ in range(token_count):
                obj = self.random_token(tuple(context_queue)[0])
                result.append(obj[1])
                if n > 1:
                    context_queue.append(obj[1])
                    context_queue.pop(0)
            return (result)

