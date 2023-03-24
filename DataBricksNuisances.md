# DataBricks nuisances

Notebook files in ipynb format when in a cloned repo are not interpreted as notebooks.  (They render as json, as if they were text.)  In comparison notebooks that are imported to a repository using the import command in the dropdown menu are converted properly to notebooks.  

On the other hand, notebooks imported via the import command or created in DataBricks are stored in DataBricks custom format in the repo. 

So the preferred pattern is to originate all notebooks in the DataBricks UI.  External notebooks in a repository are not converted into Databricks format