OPTIONS VALIDVARNAME=V7;

/* 1. CREAREA ȘI FOLOSIREA DE FORMATE DEFINITE DE UTILIZATOR */
PROC FORMAT;
    VALUE outcome_fmt
        0 = 'Fără daună'
        1 = 'Cu daună';
    
    VALUE age_fmt
        '16-25' = 'Tineri (16-25)'
        '26-39' = 'Adulți (26-39)'
        '40-64' = 'Matur (40-64)'
        '65+'   = 'Senior (65+)';
    
    VALUE income_fmt
        'poverty'       = 'Sub prag sărăcie'
        'working class' = 'Clasa muncitoare'
        'middle class'  = 'Clasa de mijloc'
        'upper class'   = 'Clasa superioară';
RUN;

/* 2. CREAREA UNUI SET DE DATE SAS DIN FIȘIERE EXTERNE */
PROC IMPORT DATAFILE="C:\Facultate\PacheteSoftware\PacheteSoftwareSet2"
    OUT=work.asigurari_brut
    DBMS=PROC IMPORT DATAFILE
    REPLACE;
    GUESSINGROWS=MAX;
RUN;

/* 3. PROCESAREA ITERATIVĂ ȘI CONDIȚIONALĂ A DATELOR */
DATA work.asigurari_procesat;
    SET work.asigurari_brut;
    
    /* 4. CREAREA DE SUBSETURI DE DATE (Filtrare) */
    WHERE OUTCOME IS NOT MISSING AND CREDIT_SCORE IS NOT MISSING;
    
    /* 5. UTILIZAREA DE FUNCȚII SAS */
    CREDIT_SCORE_ROUND = ROUND(CREDIT_SCORE, 0.01);
    
    IF ANNUAL_MILEAGE = . THEN ANNUAL_MILEAGE = 12000;
    
    /* Procesare condițională (IF-THEN-ELSE) */
    IF CREDIT_SCORE < 0.3 THEN Categorie_Risc = 'Risc foarte ridicat';
    ELSE IF CREDIT_SCORE < 0.5 THEN Categorie_Risc = 'Risc ridicat';
    ELSE IF CREDIT_SCORE < 0.7 THEN Categorie_Risc = 'Risc mediu';
    ELSE Categorie_Risc = 'Risc scăzut';
    
    IF ANNUAL_MILEAGE < 8000 THEN Categorie_Kilometraj = 'Redus';
    ELSE IF ANNUAL_MILEAGE <= 15000 THEN Categorie_Kilometraj = 'Mediu';
    ELSE Categorie_Kilometraj = 'Ridicat';
    
    /* 7. UTILIZAREA DE MASIVE (ARRAY) + PROCESARE ITERATIVĂ (DO) */
    ARRAY vars_numerice[2] CREDIT_SCORE ANNUAL_MILEAGE;
    DO i = 1 TO DIM(vars_numerice);
        IF vars_numerice[i] ^= . THEN 
            vars_numerice[i] = ROUND(vars_numerice[i], 0.01);
    END;
    
    DROP i;
    
    FORMAT OUTCOME outcome_fmt. 
           AGE age_fmt. 
           INCOME income_fmt.;
RUN;

/* 6. COMBINAREA SETURILOR DE DATE (SQL și MERGE) */

/* Agregare via PROC SQL */
PROC SQL;
    CREATE TABLE work.statistici_varsta AS
    SELECT
        AGE,
        COUNT(*) AS Numar_Clienti,
        MEAN(OUTCOME) AS Rata_Daune FORMAT=PERCENT8.2,
        MEAN(CREDIT_SCORE) AS Credit_Score_Mediu FORMAT=8.3,
        MEAN(ANNUAL_MILEAGE) AS Kilometraj_Mediu FORMAT=COMMA10.0
    FROM work.asigurari_procesat
    GROUP BY AGE
    ORDER BY Rata_Daune DESC;
QUIT;

/* Agregare pe tip vehicul */
PROC SQL;
    CREATE TABLE work.statistici_vehicul AS
    SELECT
        VEHICLE_TYPE,
        COUNT(*) AS Numar_Clienti,
        MEAN(OUTCOME) AS Rata_Daune FORMAT=PERCENT8.2,
        SUM(OUTCOME) AS Total_Daune
    FROM work.asigurari_procesat
    GROUP BY VEHICLE_TYPE;
QUIT;

/* Sortarea datelor pentru MERGE */
PROC SORT DATA=work.asigurari_procesat; 
    BY AGE; 
RUN;

PROC SORT DATA=work.statistici_varsta; 
    BY AGE; 
RUN;

/* Data Step MERGE */
DATA work.asigurari_consolidat;
    MERGE work.asigurari_procesat(IN=a) 
          work.statistici_varsta(IN=b RENAME=(Numar_Clienti=Clienti_Grupa_Varsta));
    BY AGE;
    IF a AND b;
RUN;

/* 9. FOLOSIREA DE PROCEDURI STATISTICE */

PROC MEANS DATA=work.asigurari_consolidat N MEAN MIN MAX STD;
    CLASS Categorie_Risc;
    VAR CREDIT_SCORE ANNUAL_MILEAGE SPEEDING_VIOLATIONS PAST_ACCIDENTS;
    TITLE 'Statistici Descriptive - Variabile Cheie pe Categorii de Risc';
RUN;

PROC FREQ DATA=work.asigurari_consolidat;
    TABLES AGE VEHICLE_TYPE;
    TABLES OUTCOME * Categorie_Risc / CHISQ EXPECTED;
    TABLES AGE * VEHICLE_TYPE / NOPERCENT NOROW NOCOL;
    TITLE 'Analiză de Frecvență și Test Chi-pătrat';
RUN;

PROC CORR DATA=work.asigurari_consolidat NOSIMPLE;
    VAR CREDIT_SCORE ANNUAL_MILEAGE SPEEDING_VIOLATIONS PAST_ACCIDENTS;
    TITLE 'Matrice de Corelație - Variabile de Risc';
RUN;

/* 8. UTILIZAREA DE PROCEDURI PENTRU RAPORTARE */

PROC SORT DATA=work.statistici_varsta OUT=work.statistici_varsta_top;
    BY DESCENDING Rata_Daune;
RUN;

PROC REPORT DATA=work.statistici_varsta_top NOWD HEADLINE HEADSKIP;
    COLUMN AGE Numar_Clienti Rata_Daune Credit_Score_Mediu Kilometraj_Mediu;
    DEFINE AGE / ORDER 'Categorie Vârstă' WIDTH=20;
    DEFINE Numar_Clienti / DISPLAY 'Număr Clienți' FORMAT=COMMA8.;
    DEFINE Rata_Daune / DISPLAY 'Rată Daune (%)' FORMAT=PERCENT8.2;
    DEFINE Credit_Score_Mediu / DISPLAY 'Credit Score Mediu' FORMAT=8.3;
    DEFINE Kilometraj_Mediu / DISPLAY 'Kilometraj Mediu' FORMAT=COMMA10.0;
    TITLE 'Raport Detaliat - Profilul Clienților pe Categorii de Vârstă';
RUN;

PROC REPORT DATA=work.statistici_vehicul NOWD HEADLINE;
    COLUMN VEHICLE_TYPE Numar_Clienti Rata_Daune Total_Daune;
    DEFINE VEHICLE_TYPE / ORDER 'Tip Vehicul' WIDTH=15;
    DEFINE Numar_Clienti / DISPLAY 'Nr. Clienți' FORMAT=COMMA8.;
    DEFINE Rata_Daune / DISPLAY 'Rată Daune' FORMAT=PERCENT8.2;
    DEFINE Total_Daune / DISPLAY 'Total Daune' FORMAT=COMMA8.;
    TITLE 'Analiza Daunelor pe Tipuri de Vehicule';
RUN;

/* 10. GENERAREA DE GRAFICE */

ODS GRAPHICS ON;

PROC SGPLOT DATA=work.asigurari_procesat;
    VBAR AGE / GROUP=OUTCOME GROUPDISPLAY=CLUSTER;
    XAXIS LABEL='Categorie Vârstă';
    YAXIS LABEL='Număr Clienți';
    TITLE 'Distribuția Clienților cu/fără Daune pe Categorii de Vârstă';
RUN;

PROC SGPLOT DATA=work.asigurari_procesat;
    VBAR Categorie_Risc / GROUP=OUTCOME GROUPDISPLAY=STACK;
    XAXIS LABEL='Categorie Risc';
    YAXIS LABEL='Număr Total Clienți';
    TITLE 'Distribuția Clienților pe Categorii de Risc (stivuit cu/fără daune)';
RUN;

PROC SGPLOT DATA=work.asigurari_procesat;
    HISTOGRAM CREDIT_SCORE / BINWIDTH=0.05;
    DENSITY CREDIT_SCORE / TYPE=NORMAL;
    XAXIS LABEL='Credit Score';
    YAXIS LABEL='Frecvență';
    TITLE 'Distribuția Credit Score-ului în Portofoliu';
RUN;

PROC SGPLOT DATA=work.asigurari_procesat;
    SCATTER X=CREDIT_SCORE Y=ANNUAL_MILEAGE / GROUP=OUTCOME MARKERATTRS=(SIZE=5);
    XAXIS LABEL='Credit Score';
    YAXIS LABEL='Kilometraj Anual (mile)';
    TITLE 'Relația Credit Score - Kilometraj Anual (colorat după daună)';
RUN;

PROC SGPLOT DATA=work.asigurari_procesat;
    VBOX CREDIT_SCORE / CATEGORY=AGE;
    XAXIS LABEL='Categorie Vârstă';
    YAXIS LABEL='Credit Score';
    TITLE 'Distribuția Credit Score pe Categorii de Vârstă (Box Plot)';
RUN;

ODS GRAPHICS OFF;

/* EXTRA: Identificarea segmentelor profitabile */
PROC SQL;
    CREATE TABLE work.segmente_profitabile AS
    SELECT
        AGE,
        VEHICLE_TYPE,
        COUNT(*) AS Nr_Clienti,
        MEAN(OUTCOME) AS Rata_Daune FORMAT=PERCENT8.2,
        CASE 
            WHEN CALCULATED Rata_Daune < 0.25 THEN 'Profitabil'
            WHEN CALCULATED Rata_Daune < 0.35 THEN 'Neutru'
            ELSE 'Neprofitabil'
        END AS Profitabilitate
    FROM work.asigurari_procesat
    GROUP BY AGE, VEHICLE_TYPE
    HAVING Nr_Clienti >= 50
    ORDER BY Rata_Daune;
QUIT;

PROC REPORT DATA=work.segmente_profitabile NOWD;
    COLUMN AGE VEHICLE_TYPE Nr_Clienti Rata_Daune Profitabilitate;
    DEFINE AGE / ORDER 'Vârstă' WIDTH=15;
    DEFINE VEHICLE_TYPE / ORDER 'Tip Vehicul' WIDTH=12;
    DEFINE Nr_Clienti / DISPLAY 'Nr. Clienți' FORMAT=COMMA8.;
    DEFINE Rata_Daune / DISPLAY 'Rată Daune' FORMAT=PERCENT8.2;
    DEFINE Profitabilitate / DISPLAY 'Profitabilitate';
    TITLE 'Segmente Profitabile pentru Extindere - AutoSecure Insurance';
RUN;

TITLE;

PROC PRINT DATA=work.asigurari_consolidat(OBS=10) NOOBS;
    VAR ID AGE VEHICLE_TYPE CREDIT_SCORE OUTCOME Categorie_Risc 
        Clienti_Grupa_Varsta;
    TITLE 'Preview - Date Consolidate (primele 10 înregistrări)';
RUN;

TITLE;