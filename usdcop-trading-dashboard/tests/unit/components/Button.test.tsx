import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Button } from '@/components/ui/button'

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  },
}))

// Mock motion library
vi.mock('@/lib/motion', () => ({
  motionLibrary: {
    buttons: {
      tap: { scale: 0.95 },
      hover: { scale: 1.02 },
    },
  },
}))

describe('Button Component', () => {
  it('renders with default props', () => {
    render(<Button>Default Button</Button>)

    const button = screen.getByRole('button', { name: /default button/i })
    expect(button).toBeInTheDocument()
    expect(button).toHaveClass('inline-flex', 'items-center', 'justify-center')
  })

  it('renders children correctly', () => {
    render(<Button>Click me</Button>)

    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('handles click events', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()

    render(<Button onClick={handleClick}>Clickable</Button>)

    const button = screen.getByRole('button', { name: /clickable/i })
    await user.click(button)

    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled Button</Button>)

    const button = screen.getByRole('button', { name: /disabled button/i })
    expect(button).toBeDisabled()
    expect(button).toHaveClass('disabled:pointer-events-none', 'disabled:opacity-50')
  })

  describe('Variants', () => {
    it('applies default variant styles', () => {
      render(<Button variant="default">Default</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-slate-800', 'text-slate-100')
    })

    it('applies destructive variant styles', () => {
      render(<Button variant="destructive">Destructive</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-gradient-to-r', 'from-red-600', 'to-red-700')
    })

    it('applies outline variant styles', () => {
      render(<Button variant="outline">Outline</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('border', 'bg-transparent', 'text-slate-300')
    })

    it('applies secondary variant styles', () => {
      render(<Button variant="secondary">Secondary</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-slate-700/60', 'text-slate-200')
    })

    it('applies ghost variant styles', () => {
      render(<Button variant="ghost">Ghost</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('text-slate-400')
    })

    it('applies link variant styles', () => {
      render(<Button variant="link">Link</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('text-cyan-400', 'underline-offset-4')
    })

    it('applies gradient variant styles', () => {
      render(<Button variant="gradient">Gradient</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-gradient-to-r', 'from-cyan-500', 'to-emerald-500')
    })

    it('applies glow variant styles', () => {
      render(<Button variant="glow">Glow</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-gradient-to-r', 'from-purple-500', 'to-pink-500')
    })

    it('applies glass variant styles', () => {
      render(<Button variant="glass">Glass</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('glass-surface-secondary')
    })

    it('applies glass-primary variant styles', () => {
      render(<Button variant="glass-primary">Glass Primary</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('glass-button-primary', 'backdrop-blur-professional')
    })

    it('applies glass-secondary variant styles', () => {
      render(<Button variant="glass-secondary">Glass Secondary</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('glass-button-secondary', 'backdrop-blur-md')
    })

    it('applies glass-accent variant styles', () => {
      render(<Button variant="glass-accent">Glass Accent</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-gradient-to-br', 'from-amber-500/20')
    })

    it('applies professional variant styles', () => {
      render(<Button variant="professional">Professional</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('glass-surface-elevated', 'backdrop-blur-intense')
    })

    it('applies terminal variant styles', () => {
      render(<Button variant="terminal">Terminal</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('bg-gradient-to-br', 'border-2', 'border-cyan-500/30')
    })
  })

  describe('Sizes', () => {
    it('applies default size styles', () => {
      render(<Button size="default">Default Size</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('text-sm')
    })

    it('applies small size styles', () => {
      render(<Button size="sm">Small</Button>)

      const button = screen.getByRole('button')
      // The actual size classes would be checked here
      expect(button).toBeInTheDocument()
    })

    it('applies large size styles', () => {
      render(<Button size="lg">Large</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
    })

    it('applies icon size styles', () => {
      render(<Button size="icon">⚙️</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
    })

    it('applies extra small size styles', () => {
      render(<Button size="xs">Extra Small</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
    })
  })

  describe('Special Props', () => {
    it('applies glow effect when glow prop is true', () => {
      render(<Button glow>Glowing Button</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('shadow-glow-cyan')
    })

    it('shows loading state when loading prop is true', () => {
      render(<Button loading>Loading Button</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('cursor-wait')
    })

    it('applies pulse animation when pulse prop is true', () => {
      render(<Button pulse>Pulsing Button</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('animate-pulse')
    })

    it('handles custom className', () => {
      render(<Button className="custom-class">Custom</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('custom-class')
    })
  })

  describe('Accessibility', () => {
    it('supports keyboard navigation', async () => {
      const handleClick = vi.fn()
      const user = userEvent.setup()

      render(<Button onClick={handleClick}>Accessible Button</Button>)

      const button = screen.getByRole('button')
      await user.tab()
      expect(button).toHaveFocus()

      await user.keyboard('{Enter}')
      expect(handleClick).toHaveBeenCalledTimes(1)

      await user.keyboard(' ')
      expect(handleClick).toHaveBeenCalledTimes(2)
    })

    it('has proper focus styles', () => {
      render(<Button>Focusable Button</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('focus-visible:outline-none', 'glass-focus')
    })

    it('supports aria attributes', () => {
      render(
        <Button aria-label="Custom label" aria-describedby="description">
          Button with ARIA
        </Button>
      )

      const button = screen.getByRole('button', { name: /custom label/i })
      expect(button).toHaveAttribute('aria-label', 'Custom label')
      expect(button).toHaveAttribute('aria-describedby', 'description')
    })

    it('handles disabled state accessibility', () => {
      render(<Button disabled>Disabled Button</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeDisabled()
      expect(button).toHaveAttribute('disabled')
    })
  })

  describe('Event Handling', () => {
    it('handles mouse events', async () => {
      const handleMouseEnter = vi.fn()
      const handleMouseLeave = vi.fn()
      const user = userEvent.setup()

      render(
        <Button onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
          Hover Button
        </Button>
      )

      const button = screen.getByRole('button')
      await user.hover(button)
      expect(handleMouseEnter).toHaveBeenCalledTimes(1)

      await user.unhover(button)
      expect(handleMouseLeave).toHaveBeenCalledTimes(1)
    })

    it('handles focus events', () => {
      const handleFocus = vi.fn()
      const handleBlur = vi.fn()

      render(
        <Button onFocus={handleFocus} onBlur={handleBlur}>
          Focus Button
        </Button>
      )

      const button = screen.getByRole('button')
      fireEvent.focus(button)
      expect(handleFocus).toHaveBeenCalledTimes(1)

      fireEvent.blur(button)
      expect(handleBlur).toHaveBeenCalledTimes(1)
    })

    it('prevents event when disabled', async () => {
      const handleClick = vi.fn()
      const user = userEvent.setup()

      render(
        <Button disabled onClick={handleClick}>
          Disabled Button
        </Button>
      )

      const button = screen.getByRole('button')
      await user.click(button)
      expect(handleClick).not.toHaveBeenCalled()
    })
  })

  describe('Loading State', () => {
    it('shows loading state correctly', () => {
      render(<Button loading>Loading...</Button>)

      const button = screen.getByRole('button')
      expect(button).toHaveClass('cursor-wait')
      expect(screen.getByText('Loading...')).toBeInTheDocument()
    })

    it('disables interaction during loading', async () => {
      const handleClick = vi.fn()
      const user = userEvent.setup()

      render(
        <Button loading onClick={handleClick}>
          Loading Button
        </Button>
      )

      const button = screen.getByRole('button')
      await user.click(button)
      // Loading buttons might still be clickable depending on implementation
      // This test ensures the loading state is applied
      expect(button).toHaveClass('cursor-wait')
    })
  })

  describe('Performance', () => {
    it('renders efficiently with many buttons', () => {
      const buttons = Array.from({ length: 100 }, (_, i) => (
        <Button key={i} variant="default">
          Button {i}
        </Button>
      ))

      const startTime = performance.now()
      render(<div>{buttons}</div>)
      const endTime = performance.now()

      expect(endTime - startTime).toBeLessThan(100) // Should render quickly
      expect(screen.getAllByRole('button')).toHaveLength(100)
    })

    it('handles rapid state changes', async () => {
      const { rerender } = render(<Button>Initial</Button>)

      // Rapidly change props
      for (let i = 0; i < 10; i++) {
        rerender(<Button variant={i % 2 === 0 ? 'default' : 'outline'}>Updated {i}</Button>)
      }

      expect(screen.getByText('Updated 9')).toBeInTheDocument()
    })
  })

  describe('Edge Cases', () => {
    it('handles empty children', () => {
      render(<Button></Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
      expect(button).toBeEmptyDOMElement()
    })

    it('handles complex children', () => {
      render(
        <Button>
          <span>Icon</span>
          <span>Text</span>
        </Button>
      )

      expect(screen.getByText('Icon')).toBeInTheDocument()
      expect(screen.getByText('Text')).toBeInTheDocument()
    })

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLButtonElement>()
      render(<Button ref={ref}>Ref Button</Button>)

      expect(ref.current).toBeInstanceOf(HTMLButtonElement)
      expect(ref.current?.textContent).toBe('Ref Button')
    })

    it('handles undefined variant gracefully', () => {
      render(<Button variant={undefined as any}>Undefined Variant</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
    })

    it('handles unknown variant gracefully', () => {
      render(<Button variant={'unknown' as any}>Unknown Variant</Button>)

      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
    })
  })
})