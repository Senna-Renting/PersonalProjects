from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox as msgbox
import threading
import asyncio
import re
import os
import logging
#new get content method
from getcontentv2 import ContentHandler
#old get content method
#import getcontent

TRUNCATION = 35

class GUI:
    DATATYPES = ["FISCAAL_RDW",
            "BPM",
            "WLTP_BENZINE",
            "CO2",
            "GEWICHT",
            "TREKHAAK",
            "KLEUR",
            "ENERGIELABEL",
            "TANKINHOUD"]

    def __init__(self):
        # make the window
        self.root = Tk()
        self.root.geometry("600x700")
        self.root.title("Kenteken Manager")
        self.kentekens_indices = []

        # configure the styling
        self.add_styles()
        # add special tkinter variables
        self.make_vars()
        # add widgets to window
        self.make_widgets()
        # configure grid structure
        self.configure_grid()
        # define positions widgets on grid
        self.place_widgets()
        # add events
        self.make_events()

        # load and start the window
        self.root.mainloop()

    # initialisation functions
    def make_vars(self):
        self.entry_text = StringVar()
        self.checkvars = [IntVar(value=1) for i in range(len(self.DATATYPES))]
        self.selected_path = StringVar(value=os.getcwd()+"\kentekens.csv")
        self.progress_count = IntVar(value=0)
        self.progress_name = StringVar(value="Nog niks aan het ophalen")

    def make_widgets(self):
        # helper function
        def make_checkbox(parent, text, index, variables):
            # add widget to frame
            checkbox = ttk.Checkbutton(parent, text=text, variable=variables[index], onvalue=1, offvalue=0, style="My.TCheckbutton")
            # define position on grid
            checkbox.grid(row=index, column=0, sticky=(W))
            return checkbox

        self.frame = ttk.Frame(self.root, height=600, width=600, style='My.TFrame')
        self.title = ttk.Label(self.frame, text='Kenteken Manager', style='My.TLabel')
        self.kenteken_entry = ttk.Entry(self.frame, textvariable=self.entry_text)
        self.add_btn = ttk.Button(self.frame, text="Toevoegen", command=self.add_kenteken, style="My.TButton")
        self.kenteken_scroll = Scrollbar(self.frame, orient="vertical")
        self.kenteken_list = Text(self.frame, height=10, width=16, yscrollcommand = self.kenteken_scroll.set)
        self.kenteken_scroll.config(command=self.kenteken_list.yview)
        self.checkbox_frame = ttk.Frame(self.frame, style="My.TFrame")
        self.checkboxes = [make_checkbox(self.checkbox_frame, text, i, self.checkvars) for i,text in enumerate(self.DATATYPES)]
        self.submit_btn = ttk.Button(self.frame, text="Maak Excel bestand", command=lambda:self.async_csv(), width=21, style="My.TButton")
        self.select_folder = ttk.Button(self.frame, text="Selecteer folder", command=self.ask_directory, width=21, style="My.TButton")
        self.announce_path = ttk.Label(self.frame, text="Download path: ")
        self.select_path_text = ttk.Label(self.frame, width=TRUNCATION, textvariable=self.selected_path)


    def place_widgets(self):
        self.frame.grid(row=0, column=0, sticky=(N,S,E,W))
        self.title.grid(row=0, column=0, columnspan=2, padx=30, pady=30)
        self.kenteken_entry.grid(row=1, column=0, padx=10, sticky=(E))
        self.add_btn.grid(row=1, column=1, padx=10, sticky=(W))
        self.kenteken_list.grid(row=2, column=0, sticky=(N,S,E), pady=10, padx=10)
        self.kenteken_scroll.grid(row=2, column=1, sticky=(N,S,W), pady=10)
        self.checkbox_frame.grid(row=2, column=1, sticky=(N,W), padx=30, pady=10)
        self.submit_btn.grid(row=3, column=1, sticky=(W))
        self.select_folder.grid(row=3, column=0, sticky=(E), padx=10)
        self.announce_path.grid(row=4, column=0, sticky=(N,E))
        self.select_path_text.grid(row=4, column=1, sticky=(N,W))

    def configure_grid(self):
        self.root.grid_columnconfigure(0,weight=1)
        self.root.grid_rowconfigure(0,weight=1)

        self.frame.grid_columnconfigure(0,weight=1)
        self.frame.grid_columnconfigure(1,weight=1)
        self.frame.grid_rowconfigure(0,weight=1)
        self.frame.grid_rowconfigure(1,weight=1)
        self.frame.grid_rowconfigure(2,weight=3)
        self.frame.grid_rowconfigure(3,weight=1)
        self.frame.grid_rowconfigure(4,weight=3)
        self.frame.grid_rowconfigure(5,weight=1)
        self.frame.grid_rowconfigure(6,weight=1)

        self.checkbox_frame.grid_columnconfigure(0, weight=1)
        for i in range(len(self.DATATYPES)):
            self.checkbox_frame.grid_rowconfigure(i,weight=1)

    # bind events to widgets
    def make_events(self):
        self.kenteken_entry.bind("<Return>", lambda e:self.add_kenteken())

    def add_styles(self):
        s = ttk.Style()
        bg = '#26467f'
        color = "white"
        s.configure('My.TFrame', background=bg)
        s.configure("My.TCheckbutton", background=bg, foreground=color)
        s.configure('My.TLabel', font=('calibri',24,'bold'), background=bg, foreground=color)
        s.configure('My.TButton', background=bg, )

    # event functions
    def ask_directory(self):
        dir = filedialog.askdirectory()
        if dir != '':
            path = dir+'/kentekens.csv'
            msg = f"Opslaglocatie gewijzigd naar: {path}"
            logging.log(logging.DEBUG,msg)
            print(msg)
            self.selected_path.set(path)

    def add_kenteken(self):
        text = self.entry_text.get().upper()
        print(f"Kenteken {text} was added to the list")
        if not len(self.kenteken_list.get("1.0", "end-1c")) == 0:
            text = f"\n{text}"
        self.kenteken_list.insert("end", text)

    def _asyncio_thread(self):
        asyncio.run(self.make_csv())

    def async_csv(self):
        threading.Thread(target=self._asyncio_thread).start()

    def make_progressbar(self, amount):
        self.progress_loader = ttk.Progressbar(self.frame, length=200, orient='horizontal', mode='determinate', variable=self.progress_count, maximum=amount+1)
        self.progress_label = ttk.Label(self.frame, textvariable=self.progress_name, width=35)
        self.progress_loader.grid(row=5, column=0, columnspan=2, pady=10, sticky=(S))
        self.progress_label.grid(row=6, column=0, columnspan=2, sticky=(N))

    def reset_progressbar(self):
        self.progress_count.set(0)

    def highlight_text(self):
        text = self.kenteken_list
        count = self.progress_count.get()
        index = self.kentekens_indices[count-1]
        text.tag_add("current", f"{index+1}.0", f"{index+2}.0")
        text.tag_config("current", background="black", foreground="white")


    async def make_csv(self):
        if msgbox.askyesno("Waarschuwing", "Weet je zeker dat je een Excel bestand wilt maken van de ingevulde kentekens?"):
            kentekens_pre = self.kenteken_list.get("1.0", "end-1c").split("\n")
            pattern = re.compile(r'\w{6}')
            kentekens = [k for k in kentekens_pre if pattern.match(k)]
            self.kentekens_indices = [i for i,k in enumerate(kentekens_pre) if pattern.match(k)]
            kwargs = dict()
            for i,type in enumerate(self.DATATYPES):
                state = bool(self.checkvars[i].get())
                kwargs[type] = state
            if kentekens[0] != '':
                self.make_progressbar(len(kentekens))
                msg = f"Kentekens op te vragen: {', '.join(kentekens)}"
                print(msg)
                logging.log(logging.DEBUG,msg)
                self.kenteken_list.configure(state="disabled")
                content_handler = ContentHandler(self, **kwargs)
                content_handler.get_license_plates(kentekens)
                content_handler.to_csv(self.selected_path.get())
                self.kenteken_list.tag_delete("current")
                self.kenteken_list.configure(state="normal")
                self.reset_progressbar()
            else:
                msgbox.showinfo("Missende kentekens", "U heeft nog geen kentekens ingevuld. Vul deze eerst in!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    GUI()
