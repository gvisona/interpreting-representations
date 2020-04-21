# Original phenoprint features from the Breinig paper

# 4 nseg.0.m.majoraxis.mean       --> nuclear shape
# 12 nseg.dna.m.eccentricity.sd   --> nuclear shape
# 20 nseg.0.s.radius.max.qt.0.05  --> nuclear shape
# 8 nseg.0.m.eccentricity.mean    --> nuclear shape

# 7 nseg.dna.h.var.s2.mean        --> nuclear texture
# 13 nseg.dna.h.idm.s1.sd         --> nuclear texture
# 17 nseg.dna.h.cor.s2.sd         --> nuclear texture

# 1 n                             --> cell number

# 6 cseg.act.h.f12.s2.sd          --> cellular texture
# 15 cseg.act.h.asm.s2.mean       --> cellular texture
# 18 cseg.dnaact.b.mad.mean       --> cellular texture
# 16 cseg.dnaact.h.den.s2.sd      --> cellular texture
# 10 cseg.dnaact.b.mean.qt.0.05   --> cellular texture
# 3 cseg.act.h.cor.s1.mean        --> cellular texture
# 19 cseg.act.h.idm.s2.sd         --> cellular texture
# 14 cseg.dnaact.h.f13.s1.mean    --> cellular texture

# 9 cseg.0.s.radius.min.qt.0.05   --> cellular shape
# 5 cseg.dnaact.m.eccentricity.sd --> cellular shape
# 11 cseg.act.m.eccentricity.mean --> cellular shape
# 2 cseg.act.m.majoraxis.mean     --> cellular shape



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
phenoprint_categories = {
    "nuclear shape": 4,
    "nuclear texture": 3,
    "cell number": 1,
    "cellular texture": 8,
    "cellular shape": 4
}


# features collected from the pairwise cluster comparison using gain importance as measure

pairwise_features = ['n',  # --> cell number
                     'lcd.20NN.qt.0.01',               # --> local cell density?

                     'cseg.0.m.majoraxis.qt.0.01',            # --> cellular shape
                     'cseg.0.s.radius.mean.qt.0.95',         # --> cellular shape
                     'cseg.dnaact.m.eccentricity.sd',    # --> cellular shape

                     'cseg.act.b.mean.mean',               # --> cellular texture
                     'cseg.act.b.mean.qt.0.01',            # --> cellular texture
                     'cseg.act.b.mean.qt.0.05',           # --> cellular texture
                     'cseg.dnaact.h.asm.s2.mean',         # --> cellular texture
                     'cseg.dnaact.h.cor.s1.mean',          # --> cellular texture
                     'cseg.dnaact.h.f13.s1.mean',       # --> cellular texture

                     'nseg.0.m.majoraxis.mean',        # --> nuclear shape
                     'nseg.0.m.majoraxis.qt.0.05',  # --> nuclear shape
                     'nseg.0.s.area.mean',          # --> nuclear shape
                     'nseg.0.s.area.qt.0.99',         # --> nuclear shape
                     'nseg.0.s.radius.max.qt.0.05',     # --> nuclear shape
                     'nseg.0.s.radius.mean.mean',       # --> nuclear shape
                     'nseg.dna.m.majoraxis.qt.0.01',    # --> nuclear shape
                     'nseg.dna.m.majoraxis.qt.0.05',    # --> nuclear shape


                     'nseg.dna.b.mean.qt.0.05'         # --> nuclear texture
                     ]

pairwise_categories = {
    "cell number and density": 2,
    "cellular shape": 3,
    "cellular texture": 6,
    "nuclear shape": 8,
    "nuclear texture": 1
}



"""
Some info on the feature names https://rdrr.io/bioc/EBImage/man/computeFeatures.html
For details on Haralick texture features https://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html


General categories
.b features are related to intensity (basic)
.s features indicate sizes in pixels (shape)
.m features are related to image moments (moment)
.h are the haralick texture features (haralick)
"""

features_categories = {
#     "cell number": ['n'],
#     "local cell density": ['lcd.10NN.mean', 'lcd.15NN.mean', 
#                            'lcd.20NN.mean', 'lcd.10NN.sd', 
#                            'lcd.15NN.sd', 'lcd.20NN.sd', 
#                            'lcd.10NN.qt.0.01', 'lcd.10NN.qt.0.05', 
#                            'lcd.10NN.qt.0.95', 'lcd.10NN.qt.0.99', 
#                            'lcd.15NN.qt.0.01', 'lcd.15NN.qt.0.05', 
#                            'lcd.15NN.qt.0.95', 'lcd.15NN.qt.0.99', 
#                            'lcd.20NN.qt.0.01', 'lcd.20NN.qt.0.05', 
#                            'lcd.20NN.qt.0.95', 'lcd.20NN.qt.0.99'],
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
}

