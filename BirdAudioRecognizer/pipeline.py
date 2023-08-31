import requests
import json
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import io
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from typing import Any

M_AMOUNT = 60
H_AMOUNT = 3600

class XenoCantoAPI:
    '''Class that interacts with the Xeno-Canto API to get content (json search result)'''
    def __init__(self):
        self.site = "https://xeno-canto.org/api/2/recordings"
        self.listeners = list()
        self.url = ""
        
    def get(self, query, page=1):
        '''Perform an API request with the given query and page'''
        self.__page = page
        self.__query = query 
        self.request = f"query={self.__query}&page={self.__page}"
        self.url = f"{self.site}?{self.request}"
        print(f"Requesting url: {self.url}")
        self.json = json.loads(requests.get(self.url, allow_redirects=True).text)
        self.notify()
        return self.json

    # Method to signal listeners of a change
    def notify(self):
        for listener in self.listeners:
            listener.update(self.json)

    def addListener(self, listener):
        self.listeners.append(listener)

    def __str__(self):
        return self.url

class XenoCantoJSON:
    '''Class that extracts information from the Xeno-Canto API response'''
    def __init__(self, json_response=None):
        if json_response:
            self.response = json_response
    def update(self, json_response:str):
        self.set_json(json_response)
    def set_json(self, json_response:str):
        self.response = json_response
    def get_filenames(self):
        return [value["file-name"] for value in self.response["recordings"]]
    def get_files(self):
        return [value["file"] for value in self.response["recordings"]]
    def get_lengths(self):
        return [value["length"] for value in self.response["recordings"]]
    def get_formats(self):
        return [value["file-name"].split(".")[-1].strip().lower() for value in self.response["recordings"]]
    def get_labels(self):
        return [value["en"] for value in self.response["recordings"]]
    def get_qualities(self):
        return [value["q"] for value in self.response["recordings"]]
    def get_page(self):
        return int(self.response["page"])
    def get_pages(self):
        return int(self.response["numPages"])
    
class WebToAudioData:
    '''Class that scrapes audio data from a website page''' 
    @staticmethod
    def convert(urls: list):
        '''Read multiple audio files from list of urls'''
        return list(filter(lambda x: x != None, [WebToAudioData.__convert_single(url) for url in urls]))
    @staticmethod
    def __convert_single(url: str) :
        '''Read audio from given url'''
        try:
            print(f"Requesting audio: {url}")
            response = requests.get(url).content
            return sf.read(io.BytesIO(response))
        except sf.LibsndfileError as e:
            print(f"Couldn't convert url to audiofile: {str(e)}")
        
class AudioToSpectrogram:
    '''Class that converts audio to a spectrogram'''
    @staticmethod
    def convert(fragments: list, max_size: int =-1):
        '''Convert a list of audio fragments into a spectrogram'''
        print(f"Converting Audiofiles to Spectrogram")
        return [AudioToSpectrogram.__convert_single(fragment, max_size = max_size) for fragment in fragments]
    @staticmethod
    def __convert_single(fragment: np.ndarray, max_size: int =-1):
        '''Convert single audio fragment into a spectrogram'''
        data = fragment[0]
        if len(fragment[0].shape) > 1:
            data = fragment[0][:,0]
        
        S = librosa.feature.melspectrogram(y=data, sr=fragment[1], fmin=1000)
        db = librosa.power_to_db(S, ref=np.max)

        if max_size > db.shape[1]:
            max_size = db.shape[1]
        return db[:,:max_size]
    
class SpectrogramViewer:
    '''Class that visualizes a spectrogram'''
    def __init__(self, spectrogram: np.ndarray, title: str = "Melspectrogram"):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time', ax=ax, fmin=1000)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

class DatabaseRetriever:
    '''Generic class that performs sophisticated requests to the Xeno-Canto API, 
       and that can store these results'''
    def __init__(self, db_name: str = "BirdDataset"):
        self.db_name = db_name
        self.api = XenoCantoAPI()
        self.reader = XenoCantoJSON()
        self.last_checked = 0
        self.api.addListener(self.reader)

    # Get requested birdname from XenoCanto site, where conditions (amount, max_secs) on search are met    
    def get(self, birdname: str, amount: int = 30, 
            max_secs: int = 60, min_secs: int = 5, 
            max_size: int = -1, store: bool = True):
        '''Get a single bird species set of spectrograms using pre-specified search parameters'''
        # Get the requested amount of files that match the conditions
        page = 1
        self.api.get(birdname, page=page)
        tot_pages = self.reader.get_pages()
        
        files = list()
        while amount > 0 and page <= tot_pages:
            if page > 1:
                self.api.get(birdname, page=page)
            cur_files,amount = self.__filter_search(amount, max_secs, min_secs)
            files.extend(cur_files)
            page += 1
            

        # Convert the filtered files into spectrograms
        audiodata = WebToAudioData.convert(files)
        specs = AudioToSpectrogram.convert(audiodata, max_size)
        # Standardize the spectrograms
        norm_specs = normSpectrograms(specs)
        # Store retrieved data
        labels = [birdname for i in range(len(specs))]
        if store:
            self.__save(norm_specs, labels)
        # Return data
        return norm_specs, labels
    
    # Helper method for get() method
    def __filter_search(self, amount: int, max_secs: int, min_secs: int):
        '''Filter results so that they meet the given conditions (search parameters)'''
        # filters used to restrict our search
        conditions = lambda x: x <= max_secs and x >= min_secs and amount > 0
        ensure_quality = lambda x: x == "A"
        # data used for the search
        files = self.reader.get_files()
        qualities = self.reader.get_qualities()
        seconds = map(stringToSeconds, self.reader.get_lengths())
        # apply the search
        result = list()
        for i,second in enumerate(seconds):
            if conditions(second) and ensure_quality(qualities[i]):
                result.append(files[i])
                amount -= 1 
        return result, amount

    # Get multiple birdnames at the same time
    def get_multiple(self, birdnames: list, amount: int = 30, 
                     max_secs: int = 60, min_secs: int = 5, 
                     max_size: int = -1, store: bool = True):
        '''Get and optionally save spectrograms of multiple bird species, 
           with specified search parameters'''
        # Get data
        specs = list()
        labels = list()
        for name in birdnames:
            spec, label = self.get(name, amount, max_secs, min_secs, max_size, store=False)
            specs.extend(spec)
            labels.extend(label)
        # Store retrieved data
        if store:
            self.__save(specs, labels)
        # Return data
        return specs, np.array(labels)
    
    # Helper method for get_multiple() method
    def __save(self, specs: list, labels: list):
        '''Safe spectrograms and labels in a .pkl file'''
        dfs = list()
        for i, spec in enumerate(specs):
            df = pd.DataFrame(spec)
            df.columns = [f"{i}-{labels[i]}" for j in range(len(df.columns))]
            dfs.append(df)
        pd.concat(dfs, axis=1).to_pickle(f"{self.db_name}.pkl")

def stringToSeconds(string: str):
    '''Convert hh:mm:ss string format to seconds'''
    pattern: Any = re.compile("((\d{1,2}):)?(\d{1,2}):(\d{1,2})")
    
    if pattern is not None: 
        groups = pattern.search(string).groups()
    else:
        groups = (0,0,0,0) # Use zero minute results as output
    mins = int(groups[2])
    secs = int(groups[3])
    hours = int(groups[1])

    return hours*H_AMOUNT+mins*M_AMOUNT+secs

def secondsToString(secs: int):
    '''Convert seconds to string hh:mm:ss format'''
    hours = int(secs/H_AMOUNT)
    secs -= hours*H_AMOUNT
    mins = int(secs/M_AMOUNT)
    secs -= mins*M_AMOUNT
    return f"{hours}:{mins}:{secs}"
    
def normSpectrogram(spec: np.ndarray):
    '''Standardize an ndarray of values'''
    flat_spec = spec.flatten()
    mean = flat_spec.mean()
    std = flat_spec.std()
    return (spec-mean) / std

def normSpectrograms(specs: list):
    '''Convert a list of ndarrays into a standardized list of ndarrays'''
    return [normSpectrogram(spec) for spec in specs]

if __name__ == "__main__":
    # Mention what data has to be collected here!
    birds = {"Roodborst":"Erithacus rubecula", 
                 "Merel":"Turdus merula", 
                 "Vink":"Fringilla coelebs", 
                 "Koolmees":"Parus major",
                 "Heggenmus":"Prunella modularis",
                 "Pimpelmees":"Cyanistes caeruleus", 
                 "Huismus":"Passer domesticus",
                 "Winterkoning":"Troglodytes troglodytes",
                 "Ekster":"Pica pica",
                 "Gaai":"Garrulus glandarius",
                 "Boomklever":"Sitta europaea",
                 "Boomkruiper":"Certhia brachydactyla",
                 "Kauw":"Coloeus monedula",
                 "Kokmeeuw":"Chroicocephalus ridibundus",
                 "Houtduif":"Columba palumbus",
                 "Zwarte kraai":"Corvus corone",
                 "Grote bonte specht":"Dendrocopos major",
                 "Kleine bonte specht":"Dryobates minor",
                 "Groenling":"Chloris chloris",
                 "Halsbandparkiet":"Psittacula krameri",
                 "Spreeuw":"Sturnus vulgaris",
                 "Kievit":"Vanellus vanellus",
                 "Wilde eend":"Anas platyrhynchos",
                 "Turkse tortel":"Streptopelia decaocto",
                 "Zanglijster":"Turdus philomelos",
                 "Stadsduif":"Columba livia"}
    retriever = DatabaseRetriever(db_name="25DutchBirdsDataset")
    retriever.get_multiple(list(birds.values()), amount=50, min_secs=10, max_secs=35, max_size=1000)

    # Check normSpec functions
    # spec = np.array([[1,2,5,3],[0.5,10,8,2],[9,9,7,7]])
    # print(normSpectrogram(spec))


