import faiss


def get_invlist(invlists, l):
    """ returns the inverted lists content. 
    That the data is *not* copied: if the inverted index is deallocated or changes, accessing the array may crash.
    To avoid this, just clone the output arrays on output. """
    ls = invlists.list_size(l)
    list_ids = faiss.rev_swig_ptr(invlists.get_ids(l), ls)
    list_codes = faiss.rev_swig_ptr(invlists.get_codes(l), ls * invlists.code_size).reshape(ls, invlists.code_size)
    return list_ids, list_codes