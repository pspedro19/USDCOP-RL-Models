import { Link, useLocation } from 'react-router-dom'
import { User, Shield, Bell } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const settingsTabs = [
  { name: 'Profile', href: '/settings/profile', icon: User, description: 'Manage your account info' },
  { name: 'Security', href: '/settings/security', icon: Shield, description: 'Password and 2FA settings' },
  { name: 'Notifications', href: '/settings/notifications', icon: Bell, description: 'Email and push notifications' },
]

export function SettingsPage() {
  const location = useLocation()

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Manage your account settings and preferences"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {settingsTabs.map((tab) => {
          const isActive = location.pathname === tab.href
          return (
            <Link key={tab.name} to={tab.href}>
              <Card
                variant={isActive ? 'success' : 'bordered'}
                hover
                className="h-full"
              >
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className={cn(
                      'h-12 w-12 rounded-xl flex items-center justify-center',
                      isActive ? 'bg-terminal-accent/20' : 'bg-slate-800'
                    )}>
                      <tab.icon className={cn(
                        'h-6 w-6',
                        isActive ? 'text-terminal-accent' : 'text-text-muted'
                      )} />
                    </div>
                    <div>
                      <h3 className="font-semibold text-text-primary">{tab.name}</h3>
                      <p className="text-sm text-text-muted">{tab.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
