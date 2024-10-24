import pyttsx3
import threading
import queue

class SpeechSynthesizer:
    def __init__(self):
       
        self.engine = pyttsx3.init()  # Inizializzazione del motore di sintesi vocale
        
        try: # Impostazione della voce in italiano se disponibile
            voices = self.engine.getProperty('voices')
            italian_voice = next((voice for voice in voices if 'italian' in voice.languages), None)
            if italian_voice:
                self.engine.setProperty('voice', italian_voice.id)
        except:
            pass  
        
        self.engine.setProperty('rate', 200) # Da qua posso impostare la velocità del parlato
        self.speech_queue = queue.Queue()
        
        self.is_running = True
        
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()  # Avvio del thread per la gestione della sintesi vocale

    def _process_speech_queue(self): #Thread worker che processa la coda dei messaggi da pronunciare
        
        while self.is_running:
            try:
                # Attendo un nuovo messaggio dalla coda
                text = self.speech_queue.get(timeout=0.1)
                if text:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Errore durante la sintesi vocale: {e}")

    def speak_letter(self, letter):
        if letter == " ":
            self.speech_queue.put("spazio")
        else:
            self.speech_queue.put(letter)

    def speak_phrase(self, phrase):
        if phrase and not phrase.isspace():
            self.speech_queue.put(phrase)

    def cleanup(self): # Pulizia delle risorse
        self.is_running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join()
        self.engine.stop()

def create_synthesizer(): #Factory function per creare un'istanza del sintetizzatore 
    return SpeechSynthesizer()