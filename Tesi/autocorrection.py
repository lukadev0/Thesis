from spellchecker import SpellChecker
from typing import List, Tuple

class AutoCorrector: # Inizializzo il sistema di autocorrezione usando SpellChecker con il dizionario italiano
        
    def __init__(self):
        try:
            self.spell = SpellChecker(language='it')
            
            nomi_propri = {
                "andrea", "marco", "giuseppe", "giovanni", "antonio",
                "francesco", "luigi", "roberto", "stefano", "paolo",
                "alessandro", "luca", "mario", "giorgio", "vincenzo",
                "davide", "alberto", "riccardo", "filippo", "daniele",
                "michele", "salvatore", "lorenzo", "simone", "nicola",
                "emanuele","christian",
                
                "maria", "anna", "lucia", "sofia", "giulia",
                "sara", "laura", "valentina", "chiara", "francesca",
                "elena", "martina", "alessandra", "gabriella", "rosa",
                "beatrice", "elisa", "alice", "silvia", "monica",
                "roberta", "paola", "cristina", "federica", "claudia"
            }
            
            capoluoghi = {
                "roma", "milano", "napoli", "torino", "palermo",
                "genova", "bologna", "firenze", "bari", "venezia",
                "verona", "padova", "trieste", "taranto", "brescia",
                "parma", "modena", "reggio", "perugia", "cagliari"
            }

            articoli = {'il', 'lo', 'la', 'i', 'gli', 'le'}
            
            preposizioni = {'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'}
            
            # Aggiungo le parole al dizionario
            for parola in nomi_propri | capoluoghi | articoli | preposizioni:
                self.spell.word_frequency.add(parola)
                
        except Exception as e:
            print(f"Errore nell'inizializzazione dello SpellChecker: {e}")
            raise

    def find_closest_word(self, word: str) -> Tuple[str, float]:

        # Se la parola è vuota o è uno spazio la ritorno com'è
        if not word or word.isspace():
            return word, 1.0

        # Se la parola è già corretta non c'è bisogno che la modifico ulteriormente
        if self.spell.known([word]):
            return word, 1.0

        # Trova la correzione migliore
        correction = self.spell.correction(word)
        
        if correction is None or correction == word:
            return word, 0.0

        # Calcolo un punteggio di confidenza basato sulla dimensione dell'insieme di candidati, più è piccolo l'insieme più sono confidente della correzione
        candidates = self.spell.candidates(word)
        if candidates:
           
            confidence = 1.0 / len(candidates)   
            confidence = 0.5 + (confidence * 0.5) # Normalizziamo il punteggio tra 0.5 e 1.0
        else:
            confidence = 0.0

        return correction, confidence

    def correct_phrase(self, phrase: str) -> Tuple[str, List[Tuple[str, str, float]], dict]: #correzione in caso di frase totalmente sbagliata

        if not phrase or phrase.isspace():
            return phrase, [], {}

        words = phrase.split()
        corrections = []
        corrected_words = []
        all_candidates = {}
        
        for word in words:
            # Salta la correzione per parole molto corte o spazi
            if len(word) <= 2 or word.isspace():
                corrected_words.append(word)
                continue
                
            corrected_word, confidence = self.find_closest_word(word)
            
            # Preserva il formato originale (maiuscolo/minuscolo)
            if word.isupper():
                corrected_word = corrected_word.upper()
            elif word[0].isupper():
                corrected_word = corrected_word.capitalize()
                
            corrected_words.append(corrected_word)
            
            # Salva le correzioni effettuate e i candidati
            if word.lower() != corrected_word.lower():
                corrections.append((word, corrected_word, confidence))
                all_candidates[word] = list(self.spell.candidates(word))
        
        return ' '.join(corrected_words), corrections, all_candidates

    
    def add_words(self, words: List[str]): # aggiunta di parole nuove nel dizionario
        for word in words:
            self.spell.word_frequency.add(word)

    def cleanup(self):
        try:
            # Pulizia del dizionario
            self.spell.word_frequency.dictionary = {}
            
            # Rimozione delle referenze
            self.spell = None
            self.phonetic_rules = None
            self.common_errors = None
            self.word_patterns = None
        except Exception as e:
            print(f"Errore durante il cleanup dell'AutoCorrector: {e}")

def create_autocorrector() -> AutoCorrector: #creazione dell'istanza di autocorrector
    return AutoCorrector()