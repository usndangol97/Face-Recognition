"""
-- Created by Pravesh Budhathoki
-- Created on 2022-04-27 
"""
import logging
import os
import pickle
import threading
import time
from traceback import print_exc
import faiss
import numpy as np


class TestLogger:
    _logger = None
    _initialized = False

    def __init__(self, name="root"):
        TestLogger._logger = logging.getLogger(name=name)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        cls._logger.error(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg, *args, **kwargs):
        cls._logger.info(msg, *args, **kwargs)

    @classmethod
    def debug(cls, msg, *args, **kwargs):
        cls._logger.debug(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg, *args, **kwargs):
        cls._logger.warning(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg, *args, exc_info=True, **kwargs):
        cls._logger.exception(msg, *args, exc_info=exc_info, **kwargs)


class FaissIndexer(object):
    def __init__(self, dim):
        self.dim = dim
        self.base_indexer = None
        self.indexer = None

    def build(self):
        raise NotImplementedError

    def enroll(self, *args, **kwargs):
        raise NotImplementedError


class Search(object):
    def search(self, *args, **kwargs):
        raise NotImplementedError


class FaceIndexer(FaissIndexer, Search):
    def __init__(self, model_path, dim=512, load=True, max_samples_per_class=100):
        super(FaceIndexer, self).__init__(dim)
        self.logger = TestLogger()
        self.max_samples_per_class = max_samples_per_class
        self.id_to_uid = {}
        self.is_trained = False
        self.model_path = model_path
        self.index_filename = os.path.join(model_path, "indexer.bin")
        self.meta_filename = os.path.join(model_path, "indexer.meta")
        if self.index_filename is None:
            os.mkdir(os.path.join(self.model_path, "indexer.bin"))
        if self.meta_filename is None:
            os.mkdir(os.path.join(self.model_path, "indexer.meta"))
        
        self.dim = dim
        # self.lock = threading.Lock()
        self.indexer = None
        if load:
            self.load()

    def build(self):
        self.base_indexer = faiss.IndexFlatIP(int(self.dim))
        self.indexer = faiss.IndexIDMap(self.base_indexer)

    def enroll(self, embds_array=None, user_ids=None):
        # self.lock.acquire()
        try:
            if embds_array is None:
                self.logger.warning("Empty embeddings array")
                return None

            if self.indexer is None:
                self.logger.info("Creating indexer...")
                self.build()
            else:
                self.logger.info("Updating indexer...")

            store = embds_array
            if user_ids is None:
                user_ids = list(store.keys())
            else:
                if isinstance(user_ids, str):
                    user_ids = [user_ids]
            user_ids = np.array(user_ids)
            print("id2uid", self.id_to_uid)
            print("new ids", user_ids)
            self.logger.debug("User ids: {}".format(user_ids))
            todo_id_map = {}
            if not self.id_to_uid:
                for idx, uid in enumerate(user_ids):
                    self.id_to_uid[idx] = uid
                todo_id_map = self.id_to_uid
            else:
                idx = np.max(list(self.id_to_uid.keys())) + 1
                for uid in user_ids:
                    idx_ = self.map_kv(uid, self.id_to_uid, inverse=True, default=None)
                    if idx_ is None:
                        idx_ = idx
                        self.id_to_uid[idx_] = uid
                        idx += 1
                    todo_id_map[idx_] = uid
            print("id2uid", self.id_to_uid)
            print("todo", todo_id_map)
            for idx, uid in todo_id_map.items():
                t = time.time()
                uidx = idx * self.max_samples_per_class
                embds = list(store[uid])[:self.max_samples_per_class - 1]
                embds.append(np.mean(embds, axis=0))  # mean embedding
                embds = np.array(embds, dtype=np.float32)
                faiss.normalize_L2(embds)
                embds_ids = np.array([uidx + j for j in range(len(embds))])
                id_range = np.array(
                    list(range(idx * self.max_samples_per_class, (idx + 1) * self.max_samples_per_class)))
                self.indexer.remove_ids(id_range)
                self.indexer.add_with_ids(embds, embds_ids)
                t = time.time() - t
                logging.info("Indexed: {}, total: {}, time: {:.3f} sec.".format(uid, self.indexer.ntotal, t))
            logging.info("Done.")
            self.is_trained = self.indexer.is_trained
            self.save()
            return True
        except Exception as e:
            print_exc()
            return False
        finally:
            # self.lock.release()
            return True

    def load(self):
        if not os.path.exists(self.index_filename):
            self.logger.error("Creating model")
            return None
        try:
            print("index_filename", self.index_filename)
            temp = self.load_pickle_file(self.meta_filename)
            self.dim = temp["d"]
            self.max_samples_per_class = temp["max_samples_per_class"]
            self.id_to_uid = temp["id_to_uid"]
            self.indexer = faiss.read_index(self.index_filename)
            self.is_trained = self.indexer.is_trained
            self.logger.info("Indexer loaded from {}.".format(self.index_filename))
            print("Indexer loaded from {}".format(self.index_filename))
        except Exception as e:
            self.logger.exception(e)
        return self

    def save(self):
        print({"d": self.dim, "id_to_uid": self.id_to_uid,
               "max_samples_per_class": self.max_samples_per_class})
        self.save_to_pickle_file(self.meta_filename, {"d": self.dim, "id_to_uid": self.id_to_uid,
                                                      "max_samples_per_class": self.max_samples_per_class})
        faiss.write_index(self.indexer, self.index_filename)
        self.logger.info("Model is saved")

    def search(self, x, k):
        try:
            def m(keys):
                return [self.id_to_uid.get(key) for key in keys]

            # x = np.array(x, dtype=np.float32)
            if k == 0:
                k = 1
            x = np.array(x, dtype=np.float32)
            faiss.normalize_L2(x)
            a = self.indexer
            dists, indices = self.indexer.search(x, k=k)
            indices = np.array((indices - indices % self.max_samples_per_class) / self.max_samples_per_class).astype(
                np.int32)
            uids = np.ma.apply_along_axis(m, axis=1, arr=indices)
            return dists, indices, uids
        except Exception as e:
            print_exc()
            return None, None, None

    def map_kv(self, key, map_dict, inverse=False, default=None):
        try:
            if inverse:
                value = [k for k, v in map_dict.items() if v == key]
                if len(value) > 0:
                    return value[0]
                else:
                    return default
            else:
                return map_dict.get(key, default)
        except (KeyError, IndexError) as e:
            self.logger.exception(e)
            return default

    def save_to_pickle_file(self, filename, data):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.exception(e)
            return None

    def load_pickle_file(self,filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                return data
        except Exception as e:
            self.logger.exception(e)
            return None
