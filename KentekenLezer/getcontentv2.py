import mechanize
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import logging
import time

DATA_ATTRIBUTES = ["KENTEKEN",
                   "FISCAAL RDW", 
                   "BPM", 
                   "WLTP BENZINE", 
                   "WLTP ELEKTRISCH", 
                   "CO2", 
                   "GEWICHT", 
                   "TREKHAAK", 
                   "KLEUR", 
                   "ENERGIELABEL", 
                   "TANKINHOUD"]

class InfoRetrieval():
    
    def __init__(self):
        self.start()

    def ovi_request(self, license_plate):
        # Open Ovi.RDW
        self.browser.open('https://ovi.rdw.nl/')

        # Select the first (index zero) form
        self.browser.select_form(nr=0)
        
        # Let's search
        self.browser.form['ctl00$TopContent$txtKenteken']=license_plate.lower()
        self.browser.submit()

        # Show HTML of results
        return self.browser.response().read()
    
    def finnik_request(self, license_plate):
        # Open Ovi.RDW
        self.browser.open(f'https://finnik.nl/kenteken/{license_plate}/gratis')
        start_time = time.time()
        while True:
            document = self.browser.response().read()
            if "energieLabel" in document.decode("utf-8") or time.time()-start_time > 25:
                # Show HTML of results
                return document
            
    def start(self):
        # Browser
        br = mechanize.Browser()
        self.browser = br # set attribute browser for later access

        # Browser options
        br.set_handle_equiv(True)
        br.set_handle_gzip(True)
        br.set_handle_redirect(True)
        br.set_handle_referer(True)
        br.set_handle_robots(False)

        # Follows refresh 0, but it does not hang on refresh > 0
        br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

        # User-Agent (this is cheating, OK?)
        br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]

 
    def close(self):
        self.browser.close()
        
class InfoToDict():
    def __init__(self, used_site="both"):
        self.used_site = used_site # use 'ovi' or 'finnik' here!
        self.license_plate = ''
        self.results = dict()
    def info_to_dict(self,info,FISCAAL_RDW=True,
                               BPM=True,
                               WLTP_BENZINE=True,
                               CO2=True,
                               GEWICHT=True,
                               TREKHAAK=True, 
                               KLEUR=True, 
                               ENERGIELABEL=True,
                               TANKINHOUD=True):
        
        # Dictionary used to select and find the corresponding ovi data
        requested_data_ovi = {
            DATA_ATTRIBUTES[1] : [FISCAAL_RDW, InfoToDict._find_fiscaal_info],
            DATA_ATTRIBUTES[2] : [BPM, InfoToDict._find_bpm_info],
            DATA_ATTRIBUTES[3] : [WLTP_BENZINE, InfoToDict._find_benzine_info],
            DATA_ATTRIBUTES[4] : [WLTP_BENZINE, InfoToDict._find_elektrisch_info],
            DATA_ATTRIBUTES[5] : [CO2, InfoToDict._find_co2_info],
            DATA_ATTRIBUTES[6] : [GEWICHT, InfoToDict._find_gewicht_info],
            DATA_ATTRIBUTES[7] : [TREKHAAK, InfoToDict._find_trekhaak_info],
            DATA_ATTRIBUTES[8] : [KLEUR, InfoToDict._find_kleur_info]
        }
        # Dictionary used to select and find the corresponding finnik data
        requested_data_finnik = {
            DATA_ATTRIBUTES[9] : [ENERGIELABEL, InfoToDict._find_energie_info],
            DATA_ATTRIBUTES[10] : [TANKINHOUD, InfoToDict._find_tank_info]
        }
        # Reset results found
        self.results = dict()
        
        # Add requested data to results
        match self.used_site:
            case "ovi":
                self._info_to_list(info,**requested_data_ovi)
            case "finnik":
                self._info_to_list(info,**requested_data_finnik)
            case "both":
                self._info_to_list(info,**requested_data_ovi)
                self._info_to_list(info,**requested_data_finnik)
            case _:
                print("Request one of the following methods: ovi, finnik, both")
        return self.results
    
    # Convert the given data requests into a list 
    # Kwargs structure should look like this: {"DATA_REQUEST" : [bool, function], ...}
    # The bool determines if we want to use this data request, 
    # the function is the corresponding method we use to perform the request
    def _info_to_list(self,info,**kwargs):
        soup = BeautifulSoup(info,'html.parser')
        for key,value in kwargs.items():
            if value[0]:
                self.results[key] = value[1](soup)
    
    def _find_fiscaal_info(soup):
        str = re.sub(r"€\s", "", soup.find(id="CatalogusPrijs").text).strip()
        return str
    
    def _find_bpm_info(soup):
        str = re.sub(r"€\s", "", soup.find(id="BpmBedrag").text).strip()
        return str
    
    def _find_gewicht_info(soup):
        str = re.sub(r"kg", "", soup.find(id="MassaLedigVoertuig").text).strip()
        return str
    
    def _find_trekhaak_info(soup):
        str = re.sub(r"kg", "", soup.find(id="MaximumMassaGeremd").text).strip()
        return str
    
    def _find_kleur_info(soup):
        kleur = soup.find(id="Kleur")
        if kleur != None:
            return kleur.text.strip()
        return "Niet geregistreerd"
        
    def _find_energie_info(soup):
        text = soup.find(class_="energieLabel").text.strip()
        if text == "" or "Onbekend" in text:
            return  "Niet geregistreerd"
        return text
    
    def _find_tank_info(soup):
        text = soup.find("section", {"data-sectiontype":"TechnicalData"}).text
        pattern = re.search(r"([0-9]+)\sliter", text)
        if pattern != None:
            return pattern.groups()[0]
        return "Niet geregistreerd"
        
    def _find_benzine_info(soup):
        return InfoToDict._find_wltp_info(soup)
    
    def _find_elektrisch_info(soup):
        return InfoToDict._find_wltp_info(soup, isBenzine=False)

    # Generic function for finding wltp info, can be used to find both 'electrisch' and 'benzine' info
    def _find_wltp_info(soup, isBenzine=True):
        # Elektrische case
        token = 'Wh/km'
        # Benzine case
        if(isBenzine):
            token = 'l/100km'

        find_special = lambda tag: tag.name == "div" and token in tag.text and len(tag.text) < 50
        get_digits = lambda string: re.search(r"(\d*\.*\d*)", string).groups()[0]

        divs_list = soup.find_all(find_special)

        if(len(divs_list) == 0):
            return "Niet geregistreerd"
        res = float(get_digits(divs_list[-1].text))
        if res.is_integer():
            return int(res)
        else:
            return res 
    
    def _find_co2_info(soup):
        token = 'g/km'
        find_special = lambda tag: tag.name == "div" and token in tag.text and len(tag.text) < 50
        get_digits = lambda string: re.search(r"(\d*\.*\d*)", string).groups()[0]

        divs_list = soup.find_all(find_special)

        if(len(divs_list) == 0):
            return "Niet geregistreerd"
        res = float(get_digits(divs_list[-1].text))
        if res.is_integer():
            return int(res)
        else:
            return res 
        

class ContentHandler():
    def __init__(self, gui, FISCAAL_RDW=True,
                            BPM=True,
                            WLTP_BENZINE=True,
                            CO2=True,
                            GEWICHT=True,
                            TREKHAAK=True, 
                            KLEUR=True, 
                            ENERGIELABEL=True,
                            TANKINHOUD=True):
        # Link the GUI to the ContentHandler
        self.gui = gui
        # Initialize the columns and remove unused ones
        self.columns = np.array(DATA_ATTRIBUTES,dtype=object)
        self._set_columns(True, FISCAAL_RDW, BPM, WLTP_BENZINE, WLTP_BENZINE, 
                          CO2, GEWICHT, TREKHAAK, KLEUR, ENERGIELABEL, TANKINHOUD)
        
        # Determine if ovi or finnik are used/needed
        self.ovi_used = FISCAAL_RDW or BPM or WLTP_BENZINE or CO2 or GEWICHT or TREKHAAK or KLEUR
        self.finnik_used = ENERGIELABEL or TANKINHOUD
        
        # Create dataframe for results
        self.dataframe = pd.DataFrame(columns=self.columns)
        # Initialize InfoRetrieval object
        self.info_ret = InfoRetrieval()

    def _set_columns(self,*args):
        self.out_columns = self.columns[np.array(args)==0]
        self.columns = self.columns[np.array(args)==1]

    # Get dictionary that contains the selection of requests
    def _get_selection(self):
        selection_out = {col:False for col in self.out_columns}
        selection_in = {col:True for col in self.columns}
        selection_in.update(selection_out)
        return selection_in
        
    def get_license_plate(self, license_plate):
        ovi_to_dict = InfoToDict('ovi')
        finnik_to_dict = InfoToDict('finnik')
        output = dict()

        # Add license plate to output
        output[DATA_ATTRIBUTES[0]] = license_plate.upper()

        # Add data in request to output
        request_kwargs = self._get_selection()
        if self.ovi_used:
            request = self.info_ret.ovi_request(license_plate)
            output.update(ovi_to_dict.info_to_dict(request,request_kwargs))
        if self.finnik_used:
            request = self.info_ret.finnik_request(license_plate)
            output.update(finnik_to_dict.info_to_dict(request,request_kwargs))

        # Add retrieved values from request to the end of the dataframe
        self.dataframe.loc[len(self.dataframe)] = list(output.values())


    def get_license_plates(self, l_plate_list):
        self.info_ret.start()
        for license_plate in l_plate_list:
            self.gui_search(license_plate)
            self.get_license_plate(license_plate)
            self.gui_update()
        self.info_ret.close()
    
    def get_data(self):
        return self.dataframe
    
    def to_csv(self, path):
        self.dataframe.to_csv(path, decimal=',')
        msg = f"Kentekens opgeslagen in: {path}"
        logging.log(logging.DEBUG,msg)
        print(msg)
        self.gui_done()

    def gui_search(self,license_plate):
        self.gui.progress_name.set(f"Gegevens aan het zoeken voor: {license_plate}")
    
    def gui_update(self):
        counter = self.gui.progress_count.get()
        self.gui.progress_count.set(counter+1)
        self.gui.highlight_text()

    def gui_done(self):
        counter = self.gui.progress_count.get()
        self.gui.progress_count.set(counter+1)
        self.gui.progress_name.set("Klaar!") 

if __name__ == "__main__":
    # Use this block for testing parts of the code
    pass