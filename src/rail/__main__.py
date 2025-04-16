"""
This main module lets you run:
python -m rail
to list available pipeline stages or run one from the command line.
"""

if __name__ == "__main__":
    from rail import stages
    import ceci
    stages.import_and_attach_all()
    ceci.PipelineStage.main()

    
