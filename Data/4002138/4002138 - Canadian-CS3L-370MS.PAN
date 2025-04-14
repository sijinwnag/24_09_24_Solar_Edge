PVObject_=pvModule
  Version=8.0.9
  Flags=$0743

  PVObject_Commercial=pvCommercial
    Comment=www.csisolar.com  (China)
    Flags=$0041
    Manufacturer=CSI Solar
    Model=CS3L-370MS
    DataSource=Manufacturer ds 2025
    YearBeg=2025
    Width=1.048
    Height=1.765
    Depth=0.040
    Weight=22.400
    NPieces=100
    PriceDate=08/10/21 12:03
    Remarks, Count=5
      Str_1=Frame: Anodized aluminium alloy
      Str_2=Structure: 3.2mm Glass / EVA / Backsheet
      Str_3=Connections: Cable, T4 series
      Str_4
      Str_5
    End of Remarks
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=60
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  GRef=1000
  TRef=25.0
  PNom=370.0
  PNomTolLow=0.00
  PNomTolUp=2.70
  BifacialityFactor=0.000
  Isc=11.540
  Voc=40.80
  Imp=10.860
  Vmp=34.10
  muISC=5.77
  muVocSpec=-137.3
  muPmpReq=-0.320
  RShunt=350
  Rp_0=1400
  Rp_Exp=5.50
  RSerie=0.207
  Gamma=0.971
  muGamma=-0.0002
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=11.600
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  SandiaAMCorr=50.000

  OperPoints, list of=1 tOperPoint
    Point_1=True,1000,25.0,0.00,40.80,11.540,10.860,34.10,370.33
  End of List OperPoints
End of PVObject pvModule
