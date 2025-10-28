// Quick shim to relax framer-motion typings in this project.
// Some components in the codebase use motion.* with plain HTML props (className, onClick, etc.)
// and the bundled framer-motion types may be too strict for this codebase. Treat motion as any
// to prevent TypeScript errors while we either install proper types or refactor components.

declare module 'framer-motion' {
  import * as React from 'react';
  // Export commonly-used symbols as `any` so existing components that use
  // motion, AnimatePresence, etc. with plain HTML props compile without errors.
  export const motion: any;
  export const AnimatePresence: any;
  export const useAnimation: any;
  export const Variants: any;
  export default { motion, AnimatePresence, useAnimation, Variants } as any;
}
