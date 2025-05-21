# import pycosat
from typing import List
from pysat.solvers import Minisat22

from groundedPL.logUtils import LogUtils
from groundedPL.tseitin import TseitinTransform
from groundedPL.codificacion import ToPropositionalLogic, ToNumeric


class LogicTester:

    def __init__(self):
        self.to_lp = ToPropositionalLogic()
        # self.to_lp.debug = True
        self.tseitin = TseitinTransform()
        # self.tseitin.debug = True
        self.debug = False
        self.to_numeric = None
    
    def negate_sentence(self, sentence:str) -> str:
        '''
        Negate a sentence.
        '''
        if not isinstance(sentence, str):
            sentence = str(sentence)
        if sentence[0] != '-':
            negated_sentence = f'-{sentence}'
        else:
            negated_sentence = sentence[1:]
        return negated_sentence

    def translation_to_prover(self, sentence:str) -> str:
        '''
        Translate a sentence to prover format.
        '''
        sentence_lp = self.to_lp.parse(sentence)
        return sentence_lp
    
    def SATsolve(self, formula_lp:str) -> bool:
        print('transformando a tseitin...')
        print(len(formula_lp))
        formula_tseitin = self.tseitin.tseitin(formula_lp)
        print(len(formula_tseitin))
        print('resolviendo...')
        to_numeric = ToNumeric(formula_tseitin)
        formula_numeros = to_numeric.to_numeric(formula_tseitin)
        # res = pycosat.solve(formula_numeros)
        with Minisat22(bootstrap_with=formula_numeros) as m:
            if m.solve():
                res = m.get_model()
            else:
                res = 'UNSAT'
        self.to_numeric = to_numeric
        return res

    def check_implication(self, premisas:List[any], conclusion:any) -> bool:
        if len(premisas) == 0:
            formula = conclusion
        else:
            premisas_ = [self.to_lp.clases_no_vacias(formula) for formula in premisas]
            premisas_ = LogUtils.Ytoria(premisas_)
            formula = f'-({premisas_}->{conclusion})'
            # formula = f'({premisas_}∧{self.negate_sentence(conclusion)})'
        formula_lp = self.translation_to_prover(formula)
        res = self.SATsolve(formula_lp)
        if self.debug:
            print('Las premisas son:\n')
            for p in premisas:
                print('\t', p, end='\n\n')
            print('\nLa conclusion es:\n\n\t', conclusion)
            print(f'La fórmula a chequear es:\n\n\t{formula}')
            if res == 'UNSAT':
                print('\n¡La conclusión se sigue lógicamente de las premisas!')
            else:
                print('\n¡La conclusión NO se sigue lógicamente de las premisas')
                modelo = [self.to_numeric.literal(x) for x in res]
                modelo = [x for x in modelo if self.to_numeric.solo_atomo(x) in self.tseitin.atomos] 
                print(f'\nUn modelo es:\n\n\t{modelo}')
                modelo = [self.to_lp.modelo_lp.decodificar(x) for x in modelo]
                print(f'\nEl modelo decodificado es:\n\n\t{modelo}')
        return (res == 'UNSAT')

    def test_negacion(self, sentence1:str, sentence2:str) -> bool:
        '''
        Test negation between two sentences.
        '''
        #Test sentence1 implies -sentence2
        # sentence1_prover = self.translation_to_prover(sentence1)
        # negated_sentence2_prover = self.translation_to_prover(self.negate_sentence(sentence2))
        # premisas = [sentence1_prover]
        # conclusion = negated_sentence2_prover
        # resultado1 = self.check_implication(premisas, conclusion)
        # print('Resultado:', resultado1)
        premisas = [sentence1]
        conclusion = self.negate_sentence(sentence2)
        resultado1 = self.check_implication(premisas, conclusion)

        # Test -sentence1 implies sentence2
        # negated_sentence1_prover = self.translation_to_prover(self.negate_sentence(sentence1))
        # sentence2_prover = self.translation_to_prover(sentence2)
        # premisas = [negated_sentence1_prover]
        # conclusion = sentence2_prover
        # resultado2 = self.check_implication(premisas, conclusion)
        # print('Resultado:', resultado2)
        negated_sentence1 = self.negate_sentence(sentence1)
        premisas = [negated_sentence1]
        conclusion = sentence2
        resultado2 = self.check_implication(premisas, conclusion)

        return resultado1 and resultado2
    
    def test_equivalencia(self, sentence1:str, sentence2:str) -> bool:
        '''
        Test equivalence between two sentences.
        '''
        # Test sentence1 implies sentence2
        premisas = [sentence1]
        conclusion = sentence2
        resultado1 = self.check_implication(premisas, conclusion)

        # Test sentence2 implies sentence1
        premisas = [sentence2]
        conclusion = sentence1
        resultado2 = self.check_implication(premisas, conclusion)

        return resultado1 and resultado2
    
    def test_implicacion(self, sentence1:str, sentence2:str) -> bool:
        '''
        Test implication between two sentences.
        '''
        # Test sentence1 implies sentence2
        premisas = [sentence1]
        conclusion = sentence2
        resultado = self.check_implication(premisas, conclusion)

        return resultado 
    
    def test_implicacion_inversa(self, sentence1:str, sentence2:str) -> bool:
        '''
        Test inverse implication between two sentences.
        '''
        # Test sentence2 implies sentence1
        premisas = [sentence2]
        conclusion = sentence1
        resultado = self.check_implication(premisas, conclusion)

        return resultado