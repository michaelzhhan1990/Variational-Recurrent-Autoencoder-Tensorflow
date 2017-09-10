class Config(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)
    if not self.__dict__.get("kl_min"):
      self.__dict__.update({ "kl_min": None })
    if not self.__dict__.get("max_gradient_norm"):
      self.__dict__.update({ "max_gradient_norm": 5.0 })
    #if not self.__dict__.get("load_embeddings"):
    #  self.__dict__.update({ "load_embeddings": False })
    if not self.__dict__.get("batch_size"):
      self.__dict__.update({ "batch_size": 1 })
    if not self.__dict__.get("learning_rate"):
      self.__dict__.update({ "learning_rate": 0.001 })
    if not self.__dict__.get("anneal"):
      self.__dict__.update({ "anneal": False })
    if not self.__dict__.get("beam_size"):
      self.__dict__.update({ "beam_size": 1 })
    if self.__dict__.get("beam_size") > 1:
      raise NotImplementedError("Beam search is still under implementation.")
    if not self.__dict__.get('embedding_fr_path'):
      self.__dict__.update({'embedding_fr_path':None})
    if not self.__dict__.get('embedding_en_path'):
      self.__dict__.update({'embedding_en_path':None})

  def update(self, **entries):
    self.__dict__.update(entries)
