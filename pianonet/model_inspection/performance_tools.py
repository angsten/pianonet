





def get_performance(seed_note_array=None, model=None, num_time_steps=None):
    """
    Takes in a seed note array and generated num_timesteps of piano notes sampled
    from the model's output probabilities. A full NoteArray instance, including
    the seed note array data, is returned.

    seed_note_array: Seed data for model input. If None, silence is used as the seed
    model: The Keras trained model for generating the probabilities of new notes
    num_time_steps: How many new time steps of notes to generate using the model
    """