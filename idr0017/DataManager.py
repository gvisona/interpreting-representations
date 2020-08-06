import pickle as pkl
import numpy as np
import pandas as pd
import os
import requests
import re
from collections import OrderedDict

class DataManager:
    """
    Data managing class for the features data taken from IDR0017. 
    """

    def __init__(self):
        self.seed = 4285866
        self.data_folder = "data"
        if not os.path.exists(self.data_folder):
            raise ValueError("Data folder invalid.")
        self.output_folder = "output"
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        data_files = os.listdir(self.data_folder)
        assert ("datamatrixTransformed.pkl" in data_files), "Missing datamatrixTransformed file"
        assert ("interactions.pkl" in data_files), "Missing interactions file"

        processed_files = ["suppl_data.pkl", "data.pkl", "annotation_data.pkl"]
        for filename in processed_files:
            if filename not in data_files:
                self._process_data()
                break            

        self._data = None 
        self._suppl_data = None 
        self._interactions = None
        self._drug_df = None

        with open(os.path.join(self.data_folder, "annotation_data.pkl"), "rb") as f:
            annotation_data = pkl.load(f)
            self.features = annotation_data["features"]
            self.cell_line_df = annotation_data["cell_line_df"]    

        self.features_categories = features_categories 

    @property
    def data(self):
        if self._data is None:
            with open(os.path.join(self.data_folder, "data.pkl"), "rb") as f:
                self._data = pkl.load(f)
        return self._data
    
    @property
    def suppl_data(self):
        if self._suppl_data is None:
            with open(os.path.join(self.data_folder, "suppl_data.pkl"), "rb") as f:
                self._suppl_data = pkl.load(f)
        return self._suppl_data

    @property
    def interactions(self):
        if self._interactions is None:
            with open(os.path.join(self.data_folder, "interactions.pkl"), "rb") as f:
                self._interactions = pkl.load(f)
        return self._interactions

    @property
    def drug_df(self):
        if self._drug_df is None:
            with open(os.path.join(self.data_folder, "annotation_data.pkl"), "rb") as f:
                annotation_data = pkl.load(f)
                self._drug_df = annotation_data["drug_df"]
        return self._drug_df



    def test(self):
        return 1

    def _process_data(self):
        print("Preprocessing dataset...")
        with open(os.path.join(self.data_folder, "datamatrixTransformed.pkl"), "rb") as f:
            dmT = pkl.load(f)
        D = dmT["D"]
        drug_df, line, repl, ftr = dmT["anno"].values()
        with open(os.path.join(self.data_folder, "annotation_data.pkl"), "wb") as f:
            pkl.dump({"drug_df": drug_df, "cell_line_df": line, "features": ftr}, f, protocol=pkl.HIGHEST_PROTOCOL)

        colnames = ["drug", "cell_line", "replicate"]
        colnames.extend(ftr)

        # Reshape the datamatrix into a 2D array with a 3-part index
        data_values = []
        for i in range(D.shape[0]): # drug
            for j in range(D.shape[1]): #cell line
                for k in range(D.shape[2]): # replicate                    
                    row = [i,j,k]
                    row.extend(D[i,j,k,:])
                    data_values.append(row)
        data = pd.DataFrame(data=data_values, columns=colnames)
        with open(os.path.join(self.data_folder, "data.pkl"), "wb") as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

        SCREEN_ID = 1151
        INDEX_PAGE = "https://idr.openmicroscopy.org/webclient/?experimenter=-1"

        # create http session
        with requests.Session() as session:
            request = requests.Request('GET', INDEX_PAGE)
            prepped = session.prepare_request(request)
            response = session.send(prepped)
            if response.status_code != 200:
                response.raise_for_status()
                
        PLATES_URL = "https://idr.openmicroscopy.org/webclient/api/plates/?id={screen_id}"

        qs = {'screen_id': SCREEN_ID}
        url = PLATES_URL.format(**qs)
        plates = []
        plates_ref = {}
        for p in session.get(url).json()['plates']:
            plate_id = p['id']
            plates.append(p)
            plates_ref[p['name']] = p['id']    
                    
        line_ref = {i: full_name for i, full_name in enumerate(line["mutationDetailed"])}
        # Fixing some formatting
        line_ref[0] = "AKT1-/-_&_AKT2-/-"
        line_ref[3] = "CTNNB1_mt-/wt+"
        line_ref[7] = "PI3KCA_mt-/wt+"
        line_ref[8] = "KRAS_mt-/wt+"

        # Recovering plate IDs
        plate_ids = []
        for row in data.itertuples():
            cell_line = row[2]
            line_name = line_ref[cell_line]
            replicate = row[3]
            replicate_name = "Replicate_" + str(replicate+1)
            
            drug_id = row[1]
            plate = drug_df.iloc[drug_id]["PlateName"][-1]
            plate_name = line_name + "_LOPAC_" + "Plate_" + str(plate) + "_" + replicate_name
            plate_id = plates_ref[plate_name]
            plate_ids.append(plate_id)
    
        # Recovering well IDs
        WELLS_IMAGES_URL = "https://idr.openmicroscopy.org/webgateway/plate/{plate_id}/{field}/"
        wells_ref = {}
        for i, plate_id in enumerate(np.unique(plate_ids)):
            qs = {'plate_id': plate_id, 'field': 0}
            url = WELLS_IMAGES_URL.format(**qs)
            grid = session.get(url).json()
            for row in grid['grid']:
                for cell in row:
                    if cell is not None:  
                        # Find the label identifier of the well (e.g. A01)
                        a = re.search(r"Well (.+)-(.+),", cell["name"])
                        if a is None:
                            print(cell["name"])
                            break
                        b = a.groups()
                        c = b[0] + b[1]
                        if len(c) < 3:
                            c = b[0] + "0" + b[1]
                            
                        replicate = re.search(r"Replicate_(.+) \[", cell["name"]).groups()[0]
                        replicate = str(int(replicate) - 1)
                        wells_ref[str(plate_id) + "_" + replicate + "_" + c] = cell["wellId"]
                        
        well_ids = []
        for row in data.itertuples():
            idx = row[0]
            drug = row[1]
            replicate = row[3]
            well = drug_df.iloc[drug]["Well"]
            well_id = wells_ref[str(plate_ids[idx]) + "_" + str(replicate) + "_" + well]
            well_ids.append(well_id)

        image_ids = []
        images_ref = {}

        thumbnail_urls = []
        thumbnails_ref = {}

        WELLS_IMAGES_URL = "https://idr.openmicroscopy.org/webgateway/plate/{plate_id}/{field}/"
        for i, plate_id in enumerate(np.unique(plate_ids)):
            qs = {'plate_id': plate_id, 'field': 0}
            url = WELLS_IMAGES_URL.format(**qs)
            grid = session.get(url).json()
            for row in grid['grid']:
                for cell in row:
                    if cell is not None:
                        images_ref[str(plate_id)+"_"+str(cell["wellId"])] = cell["id"]
                        thumbnails_ref[str(plate_id)+"_"+str(cell["wellId"])] = cell["thumb_url"]
                        
        for pid, wid in zip(plate_ids, well_ids):
            image_ids.append(images_ref[str(pid)+"_"+str(wid)])
            thumbnail_urls.append(thumbnails_ref[str(pid)+"_"+str(wid)])
            
        url = "http://idr.openmicroscopy.org/webclient/img_detail/"
        image_details_urls = [url + str(i) for i in image_ids]
        url2 = "http://idr.openmicroscopy.org"
        thumbnail_urls = [url2 + str(u) for u in thumbnail_urls]
        url3 = "https://idr.openmicroscopy.org/webgateway/render_image/{}/0/0/"
        image_urls = [url3.format(i) for i in image_ids]

        suppl_data = pd.DataFrame()
        suppl_data["drug"] = data["drug"]
        suppl_data["cell_line"] = data["cell_line"]
        suppl_data["replicate"] = data["replicate"]
        suppl_data["plate_id"] = plate_ids
        suppl_data["well_id"] = well_ids
        suppl_data["image_id"] = image_ids
        suppl_data["image_details_url"] = image_details_urls
        suppl_data["image_url"] = image_urls
        suppl_data["thumbnail_url"] = thumbnail_urls
        suppl_data["drug_class"] = [drug_df["Class"][i] for i in data["drug"]]

        with open(os.path.join(self.data_folder, "suppl_data.pkl"), "wb") as f:
            pkl.dump(suppl_data, f)
        
        print("Preprocessing complete")

    def get_dmso_samples(self):
        dmso_indexes = []
        for row in self.drug_df.itertuples():
            #"dimethyl" in row.Name or "DMSO" in row.SecName or 
            if "ctr" in row.GeneID and "DMSO" in row.Content:
                dmso_indexes.append(row.Index)
                
        control_data = self.data[(self.data["drug"].isin(dmso_indexes))]
        return control_data

    def get_control_samples(self):
        """
        Returns the indeces of the control samples in the data dataframe
        """
        # Identify control samples
        control_data = self.get_dmso_samples()
        control_samples = control_data[(control_data["cell_line"]==4)|(control_data["cell_line"]==11)].index
        return control_samples

    def get_top_interactions(self, percentile=2):
        # Grouping interactions
        max_interactions = []
        for d, cl, r in zip(self.data["drug"], self.data["cell_line"], self.data["replicate"]):
            max_interactions.append(max(abs(self.interactions["res"][d, cl, r, :])))
        quantile = 2 # Find the top 2-percentile of interaction strength
        top_percentile = np.quantile(max_interactions, 1 - percentile/100)
        return np.where(max_interactions>=top_percentile)[0]


    def get_feature_values(self):
        return self.data.values[:,3:]







"""
Some info on the feature names https://rdrr.io/bioc/EBImage/man/computeFeatures.html
For details on Haralick texture features https://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html


General categories
.b features are related to intensity (basic)
.s features indicate sizes in pixels (shape)
.m features are related to image moments (moment)
.h are the haralick texture features (haralick)
lcd. features describe local cell density
"""

features_categories = OrderedDict({
    "cell number and density": ['n', 'lcd.10NN.mean', 'lcd.15NN.mean', 
                                   'lcd.20NN.mean', 'lcd.10NN.sd', 
                                   'lcd.15NN.sd', 'lcd.20NN.sd', 
                                   'lcd.10NN.qt.0.01', 'lcd.10NN.qt.0.05', 
                                   'lcd.10NN.qt.0.95', 'lcd.10NN.qt.0.99', 
                                   'lcd.15NN.qt.0.01', 'lcd.15NN.qt.0.05', 
                                   'lcd.15NN.qt.0.95', 'lcd.15NN.qt.0.99', 
                                   'lcd.20NN.qt.0.01', 'lcd.20NN.qt.0.05', 
                                   'lcd.20NN.qt.0.95', 'lcd.20NN.qt.0.99'],
    "nuclear shape": ['nseg.0.m.majoraxis.mean', 'nseg.dna.m.eccentricity.sd',
                      'nseg.0.s.radius.max.qt.0.05', 'nseg.0.m.eccentricity.mean',
                      'nseg.0.s.area.mean', 'nseg.0.s.perimeter.mean',
                      'nseg.0.s.radius.mean.mean', 'nseg.0.s.radius.min.mean',
                      'nseg.0.s.radius.max.mean', 'nseg.0.s.area.sd',
                      'nseg.0.s.perimeter.sd', 'nseg.0.s.radius.mean.sd',
                      'nseg.0.s.radius.min.sd', 'nseg.0.s.radius.max.sd',
                      'nseg.0.s.area.qt.0.01', 'nseg.0.s.area.qt.0.95',
                      'nseg.0.s.area.qt.0.99', 'nseg.0.s.perimeter.qt.0.99',
                      'nseg.0.s.radius.mean.qt.0.01', 'nseg.0.s.radius.mean.qt.0.05',
                      'nseg.0.s.radius.mean.qt.0.95', 'nseg.0.s.radius.mean.qt.0.99',
                      'nseg.0.s.radius.min.qt.0.01', 'nseg.0.s.radius.min.qt.0.05',
                      'nseg.0.s.radius.min.qt.0.95', 'nseg.0.s.radius.min.qt.0.99',
                      'nseg.0.s.radius.max.qt.0.01', 'nseg.0.s.radius.max.qt.0.95',
                      'nseg.0.s.radius.max.qt.0.99', 'nseg.0.m.cx.mean', 
                      'nseg.0.m.cy.mean', 'nseg.0.m.theta.mean', 
                      'nseg.dna.m.cx.mean', 'nseg.dna.m.cy.mean', 
                      'nseg.dna.m.majoraxis.mean', 'nseg.dna.m.eccentricity.mean', 
                      'nseg.dna.m.theta.mean', 'nseg.0.m.cx.sd', 'nseg.0.m.cy.sd', 
                      'nseg.0.m.majoraxis.sd', 'nseg.0.m.eccentricity.sd', 
                      'nseg.0.m.theta.sd', 'nseg.dna.m.cx.sd', 'nseg.dna.m.cy.sd', 
                      'nseg.dna.m.majoraxis.sd', 'nseg.dna.m.theta.sd', 
                      'nseg.0.m.cx.qt.0.01', 'nseg.0.m.cx.qt.0.05', 
                      'nseg.0.m.cx.qt.0.95', 'nseg.0.m.cx.qt.0.99', 
                      'nseg.0.m.cy.qt.0.01', 'nseg.0.m.cy.qt.0.05', 
                      'nseg.0.m.cy.qt.0.95', 'nseg.0.m.cy.qt.0.99', 
                      'nseg.0.m.majoraxis.qt.0.01', 'nseg.0.m.majoraxis.qt.0.05', 
                      'nseg.0.m.majoraxis.qt.0.95', 'nseg.0.m.majoraxis.qt.0.99', 
                      'nseg.0.m.eccentricity.qt.0.01', 'nseg.0.m.eccentricity.qt.0.05', 
                      'nseg.0.m.eccentricity.qt.0.95', 'nseg.0.m.eccentricity.qt.0.99', 
                      'nseg.dna.m.cx.qt.0.01', 'nseg.dna.m.cx.qt.0.05', 
                      'nseg.dna.m.cx.qt.0.95', 'nseg.dna.m.cx.qt.0.99', 
                      'nseg.dna.m.cy.qt.0.01', 'nseg.dna.m.cy.qt.0.05', 
                      'nseg.dna.m.cy.qt.0.95', 'nseg.dna.m.cy.qt.0.99', 
                      'nseg.dna.m.majoraxis.qt.0.01', 'nseg.dna.m.majoraxis.qt.0.05', 
                      'nseg.dna.m.majoraxis.qt.0.95', 'nseg.dna.m.majoraxis.qt.0.99', 
                      'nseg.dna.m.eccentricity.qt.0.01', 'nseg.dna.m.eccentricity.qt.0.05',
                      'nseg.dna.m.eccentricity.qt.0.95', 'nseg.dna.m.eccentricity.qt.0.99'
                      ],
    "nuclear texture": ['nseg.dna.h.var.s2.mean', 'nseg.dna.h.idm.s1.sd',
                        'nseg.dna.h.cor.s2.sd', 'nseg.dna.h.asm.s1.mean', 
                        'nseg.dna.h.con.s1.mean', 'nseg.dna.h.cor.s1.mean', 
                        'nseg.dna.h.var.s1.mean', 'nseg.dna.h.idm.s1.mean', 
                        'nseg.dna.h.sav.s1.mean', 'nseg.dna.h.sva.s1.mean', 
                        'nseg.dna.h.sen.s1.mean', 'nseg.dna.h.ent.s1.mean', 
                        'nseg.dna.h.dva.s1.mean', 'nseg.dna.h.den.s1.mean', 
                        'nseg.dna.h.f12.s1.mean', 'nseg.dna.h.f13.s1.mean', 
                        'nseg.dna.h.asm.s2.mean', 'nseg.dna.h.con.s2.mean', 
                        'nseg.dna.h.cor.s2.mean', 'nseg.dna.h.idm.s2.mean', 
                        'nseg.dna.h.sav.s2.mean', 'nseg.dna.h.sva.s2.mean', 
                        'nseg.dna.h.sen.s2.mean', 'nseg.dna.h.ent.s2.mean',
                        'nseg.dna.h.dva.s2.mean', 'nseg.dna.h.den.s2.mean', 
                        'nseg.dna.h.f12.s2.mean', 'nseg.dna.h.f13.s2.mean', 
                        'nseg.dna.h.asm.s1.sd', 'nseg.dna.h.con.s1.sd', 
                        'nseg.dna.h.cor.s1.sd', 'nseg.dna.h.var.s1.sd', 
                        'nseg.dna.h.sav.s1.sd', 'nseg.dna.h.sva.s1.sd', 
                        'nseg.dna.h.sen.s1.sd', 'nseg.dna.h.ent.s1.sd', 
                        'nseg.dna.h.dva.s1.sd', 'nseg.dna.h.den.s1.sd', 
                        'nseg.dna.h.f12.s1.sd', 'nseg.dna.h.f13.s1.sd', 
                        'nseg.dna.h.asm.s2.sd', 'nseg.dna.h.con.s2.sd', 
                        'nseg.dna.h.var.s2.sd', 'nseg.dna.h.idm.s2.sd', 
                        'nseg.dna.h.sav.s2.sd', 'nseg.dna.h.sva.s2.sd', 
                        'nseg.dna.h.sen.s2.sd', 'nseg.dna.h.ent.s2.sd', 
                        'nseg.dna.h.dva.s2.sd', 'nseg.dna.h.den.s2.sd', 
                        'nseg.dna.h.f12.s2.sd', 'nseg.dna.h.f13.s2.sd'],
    "nuclear pixel intensity": ['nseg.dna.b.mean.mean', 'nseg.dna.b.sd.mean', 
                          'nseg.dna.b.mad.mean', 'nseg.dna.b.mean.sd', 
                          'nseg.dna.b.sd.sd', 'nseg.dna.b.mad.sd', 
                          'nseg.dna.b.mean.qt.0.01', 'nseg.dna.b.mean.qt.0.05', 
                          'nseg.dna.b.mean.qt.0.95', 'nseg.dna.b.mean.qt.0.99'],
    "cellular shape": ['cseg.0.s.radius.min.qt.0.05', 'cseg.dnaact.m.eccentricity.sd',
                       'cseg.act.m.eccentricity.mean', 'cseg.act.m.majoraxis.mean',
                       'cseg.0.s.area.mean', 'cseg.0.s.perimeter.mean', 
                       'cseg.0.s.radius.mean.mean', 'cseg.0.s.radius.min.mean', 
                       'cseg.0.s.radius.max.mean', 'cseg.0.s.area.sd', 
                       'cseg.0.s.perimeter.sd', 'cseg.0.s.radius.mean.sd', 
                       'cseg.0.s.radius.min.sd', 'cseg.0.s.radius.max.sd', 
                       'cseg.0.s.area.qt.0.01', 'cseg.0.s.area.qt.0.05', 
                       'cseg.0.s.area.qt.0.95', 'cseg.0.s.area.qt.0.99', 
                       'cseg.0.s.perimeter.qt.0.95', 'cseg.0.s.perimeter.qt.0.99', 
                       'cseg.0.s.radius.mean.qt.0.01', 'cseg.0.s.radius.mean.qt.0.05', 
                       'cseg.0.s.radius.mean.qt.0.95', 'cseg.0.s.radius.mean.qt.0.99', 
                       'cseg.0.s.radius.min.qt.0.01', 'cseg.0.s.radius.min.qt.0.95', 
                       'cseg.0.s.radius.min.qt.0.99', 'cseg.0.s.radius.max.qt.0.01', 
                       'cseg.0.s.radius.max.qt.0.05', 'cseg.0.s.radius.max.qt.0.95', 
                       'cseg.0.s.radius.max.qt.0.99', 'cseg.0.m.cx.mean', 
                       'cseg.0.m.cy.mean', 'cseg.0.m.majoraxis.mean', 
                       'cseg.0.m.eccentricity.mean', 'cseg.0.m.theta.mean', 
                       'cseg.act.m.cx.mean', 'cseg.act.m.cy.mean', 
                       'cseg.act.m.theta.mean', 'cseg.dnaact.m.cx.mean', 
                       'cseg.dnaact.m.cy.mean', 'cseg.dnaact.m.majoraxis.mean', 
                       'cseg.dnaact.m.eccentricity.mean', 'cseg.dnaact.m.theta.mean', 
                       'cseg.0.m.cx.sd', 'cseg.0.m.cy.sd', 'cseg.0.m.majoraxis.sd', 
                       'cseg.0.m.eccentricity.sd', 'cseg.0.m.theta.sd', 
                       'cseg.act.m.cx.sd', 'cseg.act.m.cy.sd', 
                       'cseg.act.m.majoraxis.sd', 'cseg.act.m.eccentricity.sd', 
                       'cseg.act.m.theta.sd', 'cseg.dnaact.m.cx.sd', 
                       'cseg.dnaact.m.cy.sd', 'cseg.dnaact.m.majoraxis.sd', 
                       'cseg.dnaact.m.theta.sd', 'cseg.0.m.cx.qt.0.01', 
                       'cseg.0.m.cx.qt.0.05', 'cseg.0.m.cx.qt.0.95',
                       'cseg.0.m.cx.qt.0.99', 'cseg.0.m.cy.qt.0.01',
                       'cseg.0.m.cy.qt.0.05', 'cseg.0.m.cy.qt.0.95', 
                       'cseg.0.m.cy.qt.0.99', 'cseg.0.m.majoraxis.qt.0.01', 
                       'cseg.0.m.majoraxis.qt.0.05', 'cseg.0.m.majoraxis.qt.0.95',
                       'cseg.0.m.majoraxis.qt.0.99', 'cseg.0.m.eccentricity.qt.0.01', 
                       'cseg.0.m.eccentricity.qt.0.05', 'cseg.0.m.eccentricity.qt.0.95', 
                       'cseg.0.m.eccentricity.qt.0.99', 'cseg.act.m.cx.qt.0.01', 
                       'cseg.act.m.cx.qt.0.05', 'cseg.act.m.cx.qt.0.95', 
                       'cseg.act.m.cx.qt.0.99', 'cseg.act.m.cy.qt.0.01', 
                       'cseg.act.m.cy.qt.0.05', 'cseg.act.m.cy.qt.0.95', 
                       'cseg.act.m.cy.qt.0.99', 'cseg.act.m.majoraxis.qt.0.01',
                       'cseg.act.m.majoraxis.qt.0.05', 'cseg.act.m.majoraxis.qt.0.95',
                       'cseg.act.m.majoraxis.qt.0.99', 'cseg.act.m.eccentricity.qt.0.01', 
                       'cseg.act.m.eccentricity.qt.0.05', 'cseg.act.m.eccentricity.qt.0.95',
                       'cseg.act.m.eccentricity.qt.0.99', 'cseg.dnaact.m.cx.qt.0.01', 
                       'cseg.dnaact.m.cx.qt.0.05', 'cseg.dnaact.m.cx.qt.0.95', 
                       'cseg.dnaact.m.cx.qt.0.99', 'cseg.dnaact.m.cy.qt.0.01', 
                       'cseg.dnaact.m.cy.qt.0.05', 'cseg.dnaact.m.cy.qt.0.95',
                       'cseg.dnaact.m.cy.qt.0.99', 'cseg.dnaact.m.majoraxis.qt.0.95',
                       'cseg.dnaact.m.majoraxis.qt.0.99', 'cseg.dnaact.m.eccentricity.qt.0.95', 
                       'cseg.dnaact.m.eccentricity.qt.0.99'],
    "cellular texture": ['cseg.act.h.f12.s2.sd', 'cseg.act.h.asm.s2.mean',
                         'cseg.dnaact.b.mad.mean', 'cseg.dnaact.h.den.s2.sd',
                         'cseg.dnaact.b.mean.qt.0.05', 'cseg.act.h.cor.s1.mean',
                         'cseg.act.h.idm.s2.sd', 'cseg.dnaact.h.f13.s1.mean',
                        'cseg.act.h.asm.s1.mean', 'cseg.act.h.con.s1.mean', 
                         'cseg.act.h.var.s1.mean', 'cseg.act.h.idm.s1.mean',
                         'cseg.act.h.sav.s1.mean', 'cseg.act.h.sva.s1.mean', 
                         'cseg.act.h.sen.s1.mean', 'cseg.act.h.ent.s1.mean', 
                         'cseg.act.h.dva.s1.mean', 'cseg.act.h.den.s1.mean', 
                         'cseg.act.h.f12.s1.mean', 'cseg.act.h.f13.s1.mean', 
                         'cseg.act.h.con.s2.mean', 'cseg.act.h.cor.s2.mean', 
                         'cseg.act.h.var.s2.mean', 'cseg.act.h.idm.s2.mean',
                         'cseg.act.h.sav.s2.mean', 'cseg.act.h.sva.s2.mean', 
                         'cseg.act.h.sen.s2.mean', 'cseg.act.h.ent.s2.mean', 
                         'cseg.act.h.dva.s2.mean', 'cseg.act.h.den.s2.mean', 
                         'cseg.act.h.f12.s2.mean', 'cseg.act.h.f13.s2.mean',
                         'cseg.dnaact.h.asm.s1.mean', 'cseg.dnaact.h.con.s1.mean', 
                         'cseg.dnaact.h.cor.s1.mean', 'cseg.dnaact.h.var.s1.mean', 
                         'cseg.dnaact.h.idm.s1.mean', 'cseg.dnaact.h.sav.s1.mean', 
                         'cseg.dnaact.h.sva.s1.mean', 'cseg.dnaact.h.sen.s1.mean', 
                         'cseg.dnaact.h.ent.s1.mean', 'cseg.dnaact.h.dva.s1.mean',
                         'cseg.dnaact.h.den.s1.mean', 'cseg.dnaact.h.f12.s1.mean', 
                         'cseg.dnaact.h.asm.s2.mean', 'cseg.dnaact.h.con.s2.mean', 
                         'cseg.dnaact.h.cor.s2.mean', 'cseg.dnaact.h.var.s2.mean', 
                         'cseg.dnaact.h.idm.s2.mean', 'cseg.dnaact.h.sav.s2.mean', 
                         'cseg.dnaact.h.sva.s2.mean', 'cseg.dnaact.h.sen.s2.mean', 
                         'cseg.dnaact.h.ent.s2.mean', 'cseg.dnaact.h.dva.s2.mean',
                         'cseg.dnaact.h.den.s2.mean', 'cseg.dnaact.h.f12.s2.mean',
                         'cseg.dnaact.h.f13.s2.mean', 'cseg.act.h.asm.s1.sd', 
                         'cseg.act.h.con.s1.sd', 'cseg.act.h.cor.s1.sd', 
                         'cseg.act.h.var.s1.sd', 'cseg.act.h.idm.s1.sd', 
                         'cseg.act.h.sav.s1.sd', 'cseg.act.h.sva.s1.sd', 
                         'cseg.act.h.sen.s1.sd', 'cseg.act.h.ent.s1.sd', 
                         'cseg.act.h.dva.s1.sd', 'cseg.act.h.den.s1.sd', 
                         'cseg.act.h.f12.s1.sd', 'cseg.act.h.f13.s1.sd', 
                         'cseg.act.h.asm.s2.sd', 'cseg.act.h.con.s2.sd', 
                         'cseg.act.h.cor.s2.sd', 'cseg.act.h.var.s2.sd', 
                         'cseg.act.h.sav.s2.sd', 'cseg.act.h.sva.s2.sd',
                         'cseg.act.h.sen.s2.sd', 'cseg.act.h.ent.s2.sd', 
                         'cseg.act.h.dva.s2.sd', 'cseg.act.h.den.s2.sd', 
                         'cseg.act.h.f13.s2.sd', 'cseg.dnaact.h.asm.s1.sd', 
                         'cseg.dnaact.h.con.s1.sd', 'cseg.dnaact.h.cor.s1.sd',
                         'cseg.dnaact.h.var.s1.sd', 'cseg.dnaact.h.idm.s1.sd', 
                         'cseg.dnaact.h.sav.s1.sd', 'cseg.dnaact.h.sva.s1.sd', 
                         'cseg.dnaact.h.sen.s1.sd', 'cseg.dnaact.h.ent.s1.sd',
                         'cseg.dnaact.h.dva.s1.sd', 'cseg.dnaact.h.den.s1.sd', 
                         'cseg.dnaact.h.f12.s1.sd', 'cseg.dnaact.h.f13.s1.sd', 
                         'cseg.dnaact.h.asm.s2.sd', 'cseg.dnaact.h.con.s2.sd', 
                         'cseg.dnaact.h.cor.s2.sd', 'cseg.dnaact.h.var.s2.sd',
                         'cseg.dnaact.h.idm.s2.sd', 'cseg.dnaact.h.sav.s2.sd', 
                         'cseg.dnaact.h.sva.s2.sd', 'cseg.dnaact.h.sen.s2.sd', 
                         'cseg.dnaact.h.ent.s2.sd', 'cseg.dnaact.h.dva.s2.sd', 
                         'cseg.dnaact.h.f12.s2.sd', 'cseg.dnaact.h.f13.s2.sd'],
    "cellular pixel intensity": ['cseg.act.b.mean.mean', 'cseg.act.b.sd.mean', 
                           'cseg.act.b.mad.mean', 'cseg.dnaact.b.mean.mean', 
                           'cseg.dnaact.b.sd.mean', 'cseg.act.b.mean.sd',
                           'cseg.act.b.sd.sd', 'cseg.act.b.mad.sd', 
                           'cseg.dnaact.b.mean.sd', 'cseg.dnaact.b.sd.sd', 
                           'cseg.dnaact.b.mad.sd', 'cseg.act.b.mean.qt.0.01', 
                           'cseg.act.b.mean.qt.0.05', 'cseg.act.b.mean.qt.0.95', 
                           'cseg.act.b.mean.qt.0.99', 'cseg.dnaact.b.mean.qt.0.01', 
                           'cseg.dnaact.b.mean.qt.0.95', 'cseg.dnaact.b.mean.qt.0.99']
})




phenoprint_features = ["nseg.0.m.majoraxis.mean", 
                       "nseg.dna.m.eccentricity.sd",
                      "nseg.0.s.radius.max.qt.0.05",
                      "nseg.0.m.eccentricity.mean", 
                       "nseg.dna.h.var.s2.mean", 
                       "nseg.dna.h.idm.s1.sd",
                       "nseg.dna.h.cor.s2.sd",
                      "n",
                      "cseg.act.h.f12.s2.sd",
                      "cseg.act.h.asm.s2.mean",
                      "cseg.dnaact.b.mad.mean", 
                      "cseg.dnaact.h.den.s2.sd",
                      "cseg.dnaact.b.mean.qt.0.05",
                      "cseg.act.h.cor.s1.mean",
                      "cseg.act.h.idm.s2.sd",
                      "cseg.dnaact.h.f13.s1.mean",
                      "cseg.0.s.radius.min.qt.0.05",
                      "cseg.dnaact.m.eccentricity.sd",
                      "cseg.act.m.eccentricity.mean",
                      "cseg.act.m.majoraxis.mean"]