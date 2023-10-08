import numpy as np
import random

def sequence2fa(sequence, file_name):
    with open(file_name, 'w') as f:
        for item in sequence:
            f.write('>' + '\n')
            f.write(item + '\n')

class Random_generator:
    """
    A class for generating random DNA sequences based on given probabilities.

    Attributes:
        probab_dict (dict): A dictionary containing the probabilities of each nucleotide.
        output_path (str): The output file path for saving the generated sequences.
        number (int): The number of sequences to generate.
        seq_len (int): The length of each generated sequence.
    """

    def __init__(self, probab_dict=None, output_path=None, number=None, seq_len=None):
        """
        Initializes a Random_generator object.

        Args:
            probab_dict (dict, optional): A dictionary containing the probabilities of each nucleotide.
                If not provided, the default probabilities {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25} are used.
            output_path (str, optional): The output file path for saving the generated sequences.
            number (int, optional): The number of sequences to generate.
            seq_len (int, optional): The length of each generated sequence.
        """
        self.output_path = output_path
        self.number = number
        self.seq_len = seq_len

        if probab_dict is None:
            self.probab_dict = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        else:
            self.probab_dict = probab_dict

    def generate_base(self):
        """
        Generates a random nucleotide base based on the given probabilities.

        Returns:
            base (str): A randomly generated nucleotide base.
        """
        role_matrix = [
            self.probab_dict['A'],
            self.probab_dict['A'] + self.probab_dict['C'],
            self.probab_dict['A'] + self.probab_dict['C'] + self.probab_dict['G'],
            self.probab_dict['A'] + self.probab_dict['C'] + self.probab_dict['G'] + self.probab_dict['T']
        ]
        random_num = random.uniform(0, 1)
        if random_num < role_matrix[0]:
            return 'A'
        elif random_num < role_matrix[1]:
            return 'C'
        elif random_num < role_matrix[2]:
            return 'G'
        elif random_num < role_matrix[3]:
            return 'T'

    def generate(self):
        """
        Generates random DNA sequences based on the given parameters.

        Returns:
            sequence (list): A list of randomly generated DNA sequences.
        """
        sequence = []
        for i in range(self.number):
            seq_tmp = ''
            for j in range(self.seq_len):
                seq_tmp += self.generate_base()
            sequence.append(seq_tmp)
        if self.output_path is not None:
            sequence2fa(sequence, self.output_path)
        return sequence

if __name__ == '__main__':
    number = 300
    probab_dict = {'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}
    output_path = './random_109bp.fa'
    seq_len = 109
    generator = Random_generator(probab_dict = probab_dict, output_path = output_path, number = number, seq_len = seq_len)
    role_matrix = generator.generate()
