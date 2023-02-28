from .base import DocumentsManager
from .pickle import DocumentsPickle
from .sqlite import DocumentsDB
from .utils import get_documents_manager_from_extension

__all__ = [DocumentsManager, DocumentsPickle, DocumentsDB, get_documents_manager_from_extension]
