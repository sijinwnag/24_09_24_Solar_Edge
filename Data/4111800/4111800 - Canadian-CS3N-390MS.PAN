PVObject_=pvModule
  Version=8.0.9
  Flags=$0543

  PVObject_Commercial=pvCommercial
    Comment=www.csisolar.com  (China)
    Flags=$0041
    Manufacturer=CSI Solar
    Model=CS3N-390MS
    DataSource=Manufacturer ds 2025
    YearBeg=2025
    Width=1.048
    Height=1.940
    Depth=0.035
    Weight=22.500
    NPieces=100
    PriceDate=06/06/23 12:01
    Remarks, Count=5
      Str_1=Frame: Anodized aluminium alloy
      Str_2=Structure: 3.2mm Glass / EVA / Backsheet
      Str_3=Connections: Cable, T4 series
      Str_4
      Str_5
    End of Remarks
  End of PVObject pvCommercial

  Technol=mtSiPoly
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  GRef=1000
  TRef=25.0
  PNom=390.0
  PNomTolLow=0.00
  PNomTolUp=1.30
  BifacialityFactor=0.000
  Isc=11.380
  Voc=44.10
  Imp=10.600
  Vmp=36.80
  muISC=5.69
  muVocSpec=-136.0
  muPmpReq=-0.360
  RShunt=240
  Rp_0=1000
  Rp_Exp=5.50
  RSerie=0.251
  Gamma=0.949
  muGamma=-0.0004
  VMaxIEC=1000
  VMaxUL=1000
  Absorb=0.90
  ARev=3.200
  BRev=11.040
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  SandiaAMCorr=50.000

  OperPoints, list of=1 tOperPoint
    Point_1=True,1000,25.0,0.00,44.10,11.380,10.600,36.80,390.08
  End of List OperPoints
End of PVObject pvModule
