import { motion } from "framer-motion";
import { Menu, Zap } from "lucide-react";

interface MobileHeaderProps {
  onOpenDrawer: () => void;
}

export const MobileHeader = ({ onOpenDrawer }: MobileHeaderProps) => {
  return (
    <header className="flex items-center justify-between border-b border-border bg-background px-4 py-3 lg:hidden">
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 glow-primary-subtle">
          <Zap className="h-4 w-4 text-primary" />
        </div>
        <span className="text-base font-semibold tracking-tight">Toy SD</span>
      </div>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onOpenDrawer}
        className="flex h-9 w-9 items-center justify-center rounded-md border border-border hover:bg-muted"
      >
        <Menu className="h-5 w-5" />
      </motion.button>
    </header>
  );
};
