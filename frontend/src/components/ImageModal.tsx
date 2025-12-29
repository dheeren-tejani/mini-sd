import { motion, AnimatePresence } from "framer-motion";
import { X, Download } from "lucide-react";

interface ImageModalProps {
  image: string | null;
  isOpen: boolean;
  onClose: () => void;
  onDownload: () => void;
}

export const ImageModal = ({ image, isOpen, onClose, onDownload }: ImageModalProps) => {
  if (!image) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-2 sm:p-4"
          onClick={onClose}
        >
          {/* Controls container */}
          <div className="absolute right-2 top-2 sm:right-4 sm:top-4 flex items-center gap-2">
            {/* Download button */}
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex h-8 w-8 sm:h-10 sm:w-auto items-center justify-center sm:gap-2 rounded-full bg-surface-elevated border border-border sm:px-4 hover:bg-muted transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onDownload();
              }}
            >
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline text-sm font-medium">Download</span>
            </motion.button>

            {/* Close button */}
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-full bg-surface-elevated border border-border hover:bg-muted transition-colors"
              onClick={onClose}
            >
              <X className="h-4 w-4 sm:h-5 sm:w-5" />
            </motion.button>
          </div>

          {/* Image */}
          <motion.img
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            src={image}
            alt="Generated artwork - fullscreen view"
            className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
