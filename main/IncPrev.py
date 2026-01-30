from main.ANALOGY_SCIENTIFIC.IncPrevMethods_polars import IncPrev

def processBatch(batch,
                 STUDY_START_DATE,
                 STUDY_END_DATE,
                 FILENAME,
                 DEMOGRAPHY,
                 col_end_date,
                 col_index_date,
                 batchId,) -> None:
    #Get unique categories
    CATGS = list(set([sublist if isinstance(sublist, str) else item \
            for sublist in DEMOGRAPHY for item in sublist]))

    if isinstance(batch, str):
        cols = ['INDEX_DATE', 'END_DATE',].append(batch)
    else:
        batch = list(batch)
        cols = ['INDEX_DATE', 'END_DATE',] + list(batch)

    if len(CATGS) > 0:
        cols = cols + CATGS

    #Incidence
    dat_incprev = IncPrev(STUDY_END_DATE[0],
                            STUDY_START_DATE[0],
                            FILENAME,
                            batch,
                            DEMOGRAPHY,
                            cols,
                            col_end_date=col_end_date,
                            col_index_date=col_index_date,
                            verbose=True,)

    results_inc = dat_incprev.runAnalysis(inc=True, prev=False)[0]

    #Prevalence
    dat_incprev = IncPrev(STUDY_END_DATE[1],
                            STUDY_START_DATE[1],
                            FILENAME,
                            batch,
                            DEMOGRAPHY,
                            cols,
                            verbose=True,)

    results_prev = dat_incprev.runAnalysis(inc=False, prev=True)[1]

    results = tuple([results_inc, results_prev])

    for result_ in results:
        if "Prevalence" in result_.columns:
            metric = "prev"
        else:
            metric = "inc"
        result_.write_csv(f"{DIR_OUT}out_{metric}_{batchId}.csv")

    # grouped incprev options
