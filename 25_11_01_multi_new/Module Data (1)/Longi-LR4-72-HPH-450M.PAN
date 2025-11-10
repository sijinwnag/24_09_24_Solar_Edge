PVObject_=pvModule
  Version=8.0.17
  Flags=$00500743

  PVObject_Commercial=pvCommercial
    Comment=https://www.longi.com/en/
    Flags=$0041
    Manufacturer=Longi Solar
    Model=LR4-72-HPH-450M
    DataSource=Manufacturer ds 2025
    YearBeg=2025
    Width=1.038
    Height=2.094
    Depth=0.035
    Weight=24.300
    NPieces=100
    PriceDate=28/10/25 15:33
    Remarks, Count=5
      Str_1=Frame: Aluminum
      Str_2=Structure: Tempered AR glass
      Str_3=Connections: Glass-Foil, Jbox IP68, MC4 EVO2 or mateable
      Str_4=PERC Half-cut
      Str_5
    End of Remarks
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=72
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsARCoating
  GRef=1000
  TRef=25.0
  PNom=450.0
  PNomTolLow=0.00
  PNomTolUp=3.00
  BifacialityFactor=0.000
  Isc=11.600
  Voc=49.30
  Imp=10.850
  Vmp=41.50
  muISC=5.80
  muVocSpec=-130.6
  muPmpReq=-0.340
  RShunt=600
  Rp_0=2500
  Rp_Exp=5.50
  RSerie=0.248
  Gamma=0.977
  muGamma=-0.0004
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=3.200
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  SandiaAMCorr=50.000
  RelEffic200=-2.60

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=25.0,1.00000
      Point_3=45.0,0.99500
      Point_4=60.0,0.96200
      Point_5=65.0,0.93600
      Point_6=70.0,0.90300
      Point_7=75.0,0.85100
      Point_8=80.0,0.75400
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM
End of PVObject pvModule
